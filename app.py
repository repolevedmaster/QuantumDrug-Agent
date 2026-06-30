import streamlit as st
import numpy as np, requests, json, time, math, os, subprocess, tempfile, shutil
from dataclasses import dataclass, field
from typing import List, Dict, Any
import plotly.graph_objects as go

from langchain_ollama import ChatOllama
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, AllChem, QED
import quimb.tensor as qtn

try:
    from vina import Vina
    VINA_OK = shutil.which("obabel") is not None
except Exception:
    VINA_OK = False

try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    _TOK = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    _MODEL = AutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    _MODEL.eval()
    CHEMBERTA_OK = True
except Exception:
    CHEMBERTA_OK = False

LLM_MODEL = "llama3.1"

def llm():
    return ChatOllama(model=LLM_MODEL, temperature=0.3)

import re
_HANJA_RE = re.compile(r"[\u4e00-\u9fff]")

def strip_hanja(text: str) -> str:
    return _HANJA_RE.sub("", text)

def ask(prompt: str) -> str:
    system = "너는 한국어로만 답하는 어시스턴트다. 한자(중국 한자)를 절대 사용하지 마라. 모든 용어는 순수 한글로만 표기해라. 예: 病(X)→병(O), 磁場(X)→자기장(O), 认知功能(X)→인지기능(O)."
    res = llm().invoke([("system", system), ("human", prompt)]).content
    return strip_hanja(res)

# ---------------- State ----------------
@dataclass
class Molecule:
    smiles: str
    mol: Any = None
    mw: float = 0.0
    logp: float = 0.0
    qed: float = 0.0
    tpsa: float = 0.0
    hbd: int = 0
    hba: int = 0
    bond_entropy: float = 0.0
    binding_score: float = 0.0
    admet_score: float = 0.0
    clinical_score: float = 0.0
    final_score: float = 0.0

@dataclass
class PipelineState:
    disease: str
    literature: str = ""
    target_protein: str = ""
    hypothesis: str = ""
    candidates: List[Molecule] = field(default_factory=list)
    top_candidates: List[Molecule] = field(default_factory=list)
    reasoning: str = ""
    log: List[str] = field(default_factory=list)
    receptor_pdbqt: str = ""
    pdb_id: str = ""
    docking_mode: str = "heuristic"
    admet_mode: str = "descriptor"

def log(state: PipelineState, msg: str):
    state.log.append(msg)

# ---------------- Research Agent ----------------
def research_agent(state: PipelineState):
    log(state, "[Research] PubMed 검색 중...")
    try:
        r = requests.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                          params={"db": "pubmed", "term": state.disease, "retmax": 5, "retmode": "json"},
                          timeout=10)
        ids = r.json()["esearchresult"]["idlist"]
        summ = requests.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
                             params={"db": "pubmed", "id": ",".join(ids), "retmode": "json"}, timeout=10).json()
        titles = [summ["result"][i]["title"] for i in ids]
        state.literature = "\n".join(titles)
    except Exception as e:
        state.literature = f"PubMed 접근 실패: {e}"
    log(state, "[Research] 표적 단백질 추론 중...")
    state.target_protein = ask(
        f"질병 '{state.disease}'에 대한 핵심 표적 단백질을 하나만 이름으로 답해라. 추가 설명 없이 단백질명만.")
    log(state, f"[Research] 완료. 표적: {state.target_protein.strip()}")
    fetch_receptor(state)

def fetch_receptor(state: PipelineState):
    log(state, "[Research] RCSB PDB 실제 구조 검색 중...")
    try:
        query = {
            "query": {"type": "terminal", "service": "full_text",
                       "parameters": {"value": state.target_protein.strip()}},
            "return_type": "entry",
            "request_options": {"results_content_type": ["experimental"], "paginate": {"rows": 1}},
        }
        r = requests.post("https://search.rcsb.org/rcsbsearch/v2/query", json=query, timeout=15)
        pdb_id = r.json()["result_set"][0]["identifier"]
        pdb_text = requests.get(f"https://files.rcsb.org/download/{pdb_id}.pdb", timeout=15).text
        tmp = tempfile.mkdtemp()
        pdb_path = os.path.join(tmp, f"{pdb_id}.pdb")
        open(pdb_path, "w").write(pdb_text)
        state.pdb_id = pdb_id
        if VINA_OK:
            pdbqt_path = os.path.join(tmp, f"{pdb_id}.pdbqt")
            subprocess.run(["obabel", pdb_path, "-O", pdbqt_path, "-xr"], check=True,
                            capture_output=True, timeout=60)
            state.receptor_pdbqt = pdbqt_path
            log(state, f"[Research] Receptor {pdb_id} 확보 및 PDBQT 변환 완료")
        else:
            log(state, f"[Research] Receptor {pdb_id} 확보 (obabel 미설치로 PDBQT 변환 생략)")
    except Exception as e:
        log(state, f"[Research] Receptor 확보 실패, heuristic docking으로 전환: {e}")

# ---------------- Hypothesis Agent ----------------
def hypothesis_agent(state: PipelineState):
    log(state, "[Hypothesis] 치료 전략 생성 중...")
    state.hypothesis = ask(
        f"질병: {state.disease}\n표적 단백질: {state.target_protein}\n"
        f"문헌 요약: {state.literature}\n"
        "위 정보를 바탕으로 치료 전략과 작용 기전을 3문장 이내로 설명해라.")
    log(state, "[Hypothesis] 완료")

# ---------------- Generator Agent ----------------
SEED_SMILES = [
    "CC(=O)Oc1ccccc1C(=O)O", "c1ccc2[nH]c3ccccc3c2c1", "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "COc1ccc2nc(N)nc(N)c2c1", "CCN(CC)CCNC(=O)c1ccc(N)cc1", "Cc1ccc(cc1)S(=O)(=O)N",
    "Clc1ccc(cc1)C(c1ccccc1)N1CCN(CC1)CCOCC(=O)O", "CC1=CC(=O)C=CC1=O",
    "Cn1cnc2c1c(=O)n(C)c(=O)n2C", "CC(C)NCC(O)COc1cccc2ccccc12",
]

def mutate(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    try:
        idx = np.random.randint(mol.GetNumAtoms())
        emol = Chem.RWMol(mol)
        repl = np.random.choice([6, 7, 8, 9])
        emol.GetAtomWithIdx(int(idx)).SetAtomicNum(int(repl))
        new = emol.GetMol()
        Chem.SanitizeMol(new)
        return Chem.MolToSmiles(new)
    except Exception:
        return smiles

def crossover(s1: str, s2: str) -> str:
    m1, m2 = Chem.MolFromSmiles(s1), Chem.MolFromSmiles(s2)
    if m1 is None or m2 is None:
        return s1
    f1 = Chem.GetMolFrags(m1, asMols=True)
    f2 = Chem.GetMolFrags(m2, asMols=True)
    try:
        combo = Chem.CombineMols(f1[0], f2[-1])
        Chem.SanitizeMol(combo)
        return Chem.MolToSmiles(combo)
    except Exception:
        return s1

def featurize(smiles: str) -> Molecule:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Molecule(
        smiles=smiles, mol=mol,
        mw=Descriptors.MolWt(mol), logp=Crippen.MolLogP(mol),
        qed=QED.qed(mol), tpsa=Descriptors.TPSA(mol),
        hbd=Descriptors.NumHDonors(mol), hba=Descriptors.NumHAcceptors(mol))

def generator_agent(state: PipelineState, n: int = 60):
    log(state, f"[Generator] {n}개 후보 분자 생성 중...")
    pool = list(SEED_SMILES)
    while len(pool) < n:
        a, b = np.random.choice(SEED_SMILES, 2, replace=True)
        s = mutate(a) if np.random.rand() < 0.5 else crossover(a, b)
        if Chem.MolFromSmiles(s) is not None:
            pool.append(s)
    mols = [featurize(s) for s in pool]
    state.candidates = [m for m in mols if m is not None][:n]
    log(state, f"[Generator] {len(state.candidates)}개 유효 후보 생성 완료")

# ---------------- Quantum MPS Agent ----------------
def molecule_to_mps_entropy(mol: Molecule, bond_dim: int = 8) -> float:
    fp = AllChem.GetMorganFingerprintAsBitVect(mol.mol, 2, nBits=64)
    bits = np.array(fp, dtype=float).reshape(8, 8)
    bits += 1e-9
    mps = qtn.MatrixProductState.from_dense(
        (bits.reshape(-1) / np.linalg.norm(bits.reshape(-1))), dims=[2] * 6, max_bond=bond_dim)
    ent = 0.0
    try:
        for i in range(1, mps.L):
            ent += mps.entropy(i)
        ent /= max(1, mps.L - 1)
    except Exception:
        ent = float(np.std(bits))
    return float(ent)

def quantum_agent(state: PipelineState, top_k: int = 20):
    log(state, "[Quantum] Tensor Network(MPS) 변환 및 탐색 공간 압축 중...")
    for m in state.candidates:
        m.bond_entropy = molecule_to_mps_entropy(m)
    ranked = sorted(state.candidates, key=lambda m: -(m.qed * 0.6 + m.bond_entropy * 0.4))
    state.candidates = ranked[:top_k]
    log(state, f"[Quantum] 압축 완료, Top-{top_k} 후보 선정 (Entropy 기준)")

# ---------------- Docking Agent ----------------
def ligand_to_pdbqt(mol_rdkit, out_path: str) -> bool:
    try:
        m = Chem.AddHs(mol_rdkit)
        if AllChem.EmbedMolecule(m, randomSeed=42) != 0:
            return False
        AllChem.MMFFOptimizeMolecule(m)
        pdb_path = out_path.replace(".pdbqt", ".pdb")
        Chem.MolToPDBFile(m, pdb_path)
        subprocess.run(["obabel", pdb_path, "-O", out_path], check=True,
                        capture_output=True, timeout=30)
        return os.path.exists(out_path)
    except Exception:
        return False

def heuristic_binding(m: Molecule) -> float:
    return round(-(m.qed * 4 + (1 / (1 + abs(m.logp - 2.5))) * 3 + (150 - min(m.tpsa, 150)) / 50), 3)

def dock_real(state: PipelineState, m: Molecule, tmp: str) -> float:
    lig_path = os.path.join(tmp, "lig.pdbqt")
    if not ligand_to_pdbqt(m.mol, lig_path):
        return heuristic_binding(m)
    v = Vina(sf_name="vina")
    v.set_receptor(state.receptor_pdbqt)
    v.set_ligand_from_file(lig_path)
    v.compute_vina_maps(center=[0, 0, 0], box_size=[25, 25, 25])
    v.dock(exhaustiveness=8, n_poses=5)
    return round(float(v.energies(n_poses=1)[0][0]), 3)

def docking_agent(state: PipelineState):
    real = VINA_OK and bool(state.receptor_pdbqt) and os.path.exists(state.receptor_pdbqt)
    state.docking_mode = "AutoDock Vina (실제 바이너리)" if real else "heuristic (fallback)"
    log(state, f"[Docking] 결합 친화도 계산 중... 모드: {state.docking_mode}")
    tmp = tempfile.mkdtemp()
    for m in state.candidates:
        m.binding_score = dock_real(state, m, tmp) if real else heuristic_binding(m)
    shutil.rmtree(tmp, ignore_errors=True)
    state.candidates.sort(key=lambda m: m.binding_score)
    log(state, "[Docking] 완료")

# ---------------- ADMET Agent ----------------
def chemberta_score(smiles: str) -> float:
    with torch.no_grad():
        inputs = _TOK(smiles, return_tensors="pt", truncation=True, max_length=128)
        out = _MODEL(**inputs).last_hidden_state.mean(dim=1)
        return float(torch.sigmoid(out.mean()))

def admet_agent(state: PipelineState):
    state.admet_mode = "ChemBERTa (사전학습 실제 모델)" if CHEMBERTA_OK else "RDKit descriptor (fallback)"
    log(state, f"[ADMET] 약물 유사성/독성 평가 중... 모드: {state.admet_mode}")
    for m in state.candidates:
        lipinski = sum([m.mw <= 500, m.logp <= 5, m.hbd <= 5, m.hba <= 10])
        base = (lipinski / 4) * 0.6 + m.qed * 0.4
        if CHEMBERTA_OK:
            try:
                bert = chemberta_score(m.smiles)
                base = round(base * 0.5 + bert * 0.5, 3)
            except Exception:
                pass
        m.admet_score = round(base, 3)
    log(state, "[ADMET] 완료")

# ---------------- Clinical Agent ----------------
def clinical_agent(state: PipelineState):
    log(state, "[Clinical] 임상 성공 가능성 평가 중...")
    for m in state.candidates:
        m.clinical_score = round(m.qed * 0.5 + m.admet_score * 0.5, 3)
        m.final_score = round(m.clinical_score * 0.4 + m.admet_score * 0.3 + (-m.binding_score) * 0.03, 3)
    state.candidates.sort(key=lambda m: -m.final_score)
    state.top_candidates = state.candidates[:10]
    log(state, "[Clinical] 완료")

# ---------------- Reasoning Agent ----------------
def reasoning_agent(state: PipelineState):
    log(state, "[Reasoning] 최종 후보 선정 근거 생성 중...")
    top = state.top_candidates[0]
    state.reasoning = ask(
        f"질병: {state.disease}\n"
        f"표적 단백질: {state.target_protein}\n"
        f"치료 가설: {state.hypothesis}\n"
        f"선정된 분자 SMILES: {top.smiles}\n"
        f"결합점수: {top.binding_score}, ADMET점수: {top.admet_score}, 임상점수: {top.clinical_score}\n"
        f"위 치료 가설({state.target_protein} 표적 기반)과 연결하여, "
        "이 분자가 왜 최적 후보인지 과학적 근거를 4문장 이내로 설명해라. "
        "치료 가설의 작용 기전과 분자 특성이 어떻게 연결되는지 명시해라.")
    log(state, "[Reasoning] 완료")

# ---------------- Report Agent ----------------
def report_agent(state: PipelineState) -> str:
    lines = [f"# Drug Discovery Report: {state.disease}", "",
              f"## 표적 단백질\n{state.target_protein}",
              f"## Receptor 구조\nPDB ID: {state.pdb_id or 'N/A'} | Docking 모드: {state.docking_mode} | ADMET 모드: {state.admet_mode}",
              "", f"## 치료 가설\n{state.hypothesis}", "", "## Top 10 후보 분자", ""]
    for i, m in enumerate(state.top_candidates, 1):
        lines.append(f"{i}. `{m.smiles}` | MW={m.mw:.1f} LogP={m.logp:.2f} QED={m.qed:.3f} "
                     f"Binding={m.binding_score:.2f} ADMET={m.admet_score:.2f} Final={m.final_score:.3f}")
    lines += ["", "## Reasoning", state.reasoning]
    return "\n".join(lines)

# ---------------- Orchestrator ----------------
def run_pipeline(disease: str, n_candidates: int, top_k: int, status_box):
    state = PipelineState(disease=disease)
    steps = [
        ("Research", research_agent, {}),
        ("Hypothesis", hypothesis_agent, {}),
        ("Generator", generator_agent, {"n": n_candidates}),
        ("Quantum", quantum_agent, {"top_k": top_k}),
        ("Docking", docking_agent, {}),
        ("ADMET", admet_agent, {}),
        ("Clinical", clinical_agent, {}),
        ("Reasoning", reasoning_agent, {}),
    ]
    for name, fn, kwargs in steps:
        status_box.write(f"▶ {name} Agent 실행 중...")
        fn(state, **kwargs)
        status_box.write(f"✔ {name} Agent 완료")
    return state

# ---------------- UI ----------------
st.set_page_config(page_title="Quantum-Agent Drug Discovery", layout="wide")
st.title("Quantum-Agent Drug Discovery Platform")

DISEASE_LIST = [
    "Alzheimer's Disease", "Parkinson's Disease", "Type 2 Diabetes",
    "Non-small Cell Lung Cancer", "Breast Cancer", "Colorectal Cancer",
    "Rheumatoid Arthritis", "HIV/AIDS", "Tuberculosis", "Hepatitis B",
    "Chronic Kidney Disease", "Heart Failure", "Atrial Fibrillation",
    "Multiple Sclerosis", "Amyotrophic Lateral Sclerosis (ALS)",
    "Schizophrenia", "Major Depressive Disorder", "Glioblastoma",
    "Pancreatic Cancer", "Leukemia", "직접 입력",
]

with st.sidebar:
    disease_sel = st.selectbox("질병 선택", DISEASE_LIST)
    disease = st.text_input("질병명 (직접 입력)", disease_sel if disease_sel != "직접 입력" else "")
    n_candidates = st.slider("후보 분자 수", 20, 200, 60)
    top_k = st.slider("MPS 압축 Top-K", 5, 50, 20)
    run = st.button("파이프라인 실행")

if run:
    status_box = st.empty()
    log_container = st.container()
    with st.spinner("실행 중..."):
        state = run_pipeline(disease, n_candidates, top_k, log_container)
    st.session_state["state"] = state

if "state" in st.session_state:
    state = st.session_state["state"]
    tabs = st.tabs(["요약", "MPS 시각화", "후보 분자", "Reasoning", "Report"])

    with tabs[0]:
        c1, c2 = st.columns(2)
        c1.info(f"Docking: {state.docking_mode}  |  PDB: {state.pdb_id or 'N/A'}")
        c2.info(f"ADMET: {state.admet_mode}")
        st.subheader("표적 단백질")
        st.write(state.target_protein)
        st.subheader("치료 가설")
        st.write(state.hypothesis)
        if state.top_candidates:
            st.subheader("최우선 후보")
            top = state.top_candidates[0]
            st.code(top.smiles)
            cols = st.columns(4)
            cols[0].metric("QED", f"{top.qed:.3f}")
            cols[1].metric("Binding", f"{top.binding_score:.2f}")
            cols[2].metric("ADMET", f"{top.admet_score:.2f}")
            cols[3].metric("Final Score", f"{top.final_score:.3f}")
            st.caption("↑ 치료 가설의 표적 단백질에 대한 최고 점수 후보. 상세 근거는 Reasoning 탭 참조.")
        st.subheader("실행 로그")
        st.code("\n".join(state.log))

    with tabs[1]:
        entropies = [m.bond_entropy for m in state.candidates]
        scores = [m.final_score for m in state.candidates]
        fig = go.Figure(data=go.Scatter(x=entropies, y=scores, mode="markers",
                         marker=dict(size=10, color=scores, colorscale="Viridis")))
        fig.update_layout(xaxis_title="Bond Entanglement Entropy", yaxis_title="Final Score",
                           title="MPS 탐색 공간 압축 결과")
        st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        for i, m in enumerate(state.top_candidates, 1):
            st.markdown(f"**{i}. `{m.smiles}`**")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("MW", f"{m.mw:.1f}")
            c2.metric("LogP", f"{m.logp:.2f}")
            c3.metric("QED", f"{m.qed:.3f}")
            c4.metric("Binding", f"{m.binding_score:.2f}")
            c5.metric("Final", f"{m.final_score:.3f}")

    with tabs[3]:
        st.write(state.reasoning)

    with tabs[4]:
        report = report_agent(state)
        st.markdown(report)
        st.download_button("Report 다운로드 (Markdown)", report, file_name="report.md")
