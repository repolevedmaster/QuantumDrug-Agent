import streamlit as st
import numpy as np, requests, re
from dataclasses import dataclass, field
from typing import List, Dict, Any
import plotly.graph_objects as go

from langchain_ollama import ChatOllama
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, AllChem, QED
import quimb.tensor as qtn

LLM_MODEL = "llama3.1"
CHEMBL = "https://www.ebi.ac.uk/chembl/api/data"

_HANJA_RE = re.compile(r"[\u4e00-\u9fff]")
def strip_hanja(text: str) -> str:
    return _HANJA_RE.sub("", text)

def llm():
    return ChatOllama(model=LLM_MODEL, temperature=0.3)

def ask(prompt: str) -> str:
    system = "너는 한국어로만 답하는 어시스턴트다. 한자(중국 한자)를 절대 사용하지 마라."
    res = llm().invoke([("system", system), ("human", prompt)]).content
    return strip_hanja(res)

# ============== ChEMBL Live Pharmacology Resolution ==============
def chembl_get(endpoint: str, params: dict, timeout=8):
    try:
        r = requests.get(f"{CHEMBL}/{endpoint}.json", params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

def fetch_disease_drugs(disease: str, limit=20) -> List[str]:
    """질병명으로 ChEMBL drug_indication 조회 -> 승인/임상 약물 molecule_chembl_id 목록"""
    data = chembl_get("drug_indication", {"mesh_heading__icontains": disease, "limit": limit})
    if not data or not data.get("drug_indications"):
        data = chembl_get("drug_indication", {"efo_term__icontains": disease, "limit": limit})
    if not data or not data.get("drug_indications"):
        return []
    ids = [d["molecule_chembl_id"] for d in data["drug_indications"] if d.get("molecule_chembl_id")]
    return list(dict.fromkeys(ids))

def fetch_mechanisms(molecule_ids: List[str]) -> List[dict]:
    """약물 ID -> 작용기전 + 표적 ID"""
    if not molecule_ids:
        return []
    ids_str = ",".join(molecule_ids[:25])
    data = chembl_get("mechanism", {"molecule_chembl_id__in": ids_str, "limit": 50})
    if not data:
        return []
    return data.get("mechanisms", [])

def fetch_target_names(target_ids: List[str]) -> Dict[str, str]:
    if not target_ids:
        return {}
    ids_str = ",".join(list(dict.fromkeys(target_ids))[:25])
    data = chembl_get("target", {"target_chembl_id__in": ids_str, "limit": 50})
    if not data:
        return {}
    return {t["target_chembl_id"]: t.get("pref_name", "Unknown") for t in data.get("targets", [])}

def fetch_smiles(molecule_ids: List[str]) -> Dict[str, str]:
    if not molecule_ids:
        return {}
    ids_str = ",".join(molecule_ids[:25])
    data = chembl_get("molecule", {"molecule_chembl_id__in": ids_str, "limit": 50,
                                     "only": "molecule_chembl_id,molecule_structures"})
    if not data:
        return {}
    out = {}
    for m in data.get("molecules", []):
        struct = m.get("molecule_structures")
        if struct and struct.get("canonical_smiles"):
            out[m["molecule_chembl_id"]] = struct["canonical_smiles"]
    return out

def resolve_disease_pharmacology(disease: str):
    """질병명으로부터 실시간 조회한 (타겟 목록, 작용기전, 승인약물 SMILES)를 반환한다."""
    drug_ids = fetch_disease_drugs(disease)
    if not drug_ids:
        return [], [], {}
    mechanisms = fetch_mechanisms(drug_ids)
    if not mechanisms:
        return [], [], fetch_smiles(drug_ids)
    target_ids = [m["target_chembl_id"] for m in mechanisms if m.get("target_chembl_id")]
    target_names = fetch_target_names(target_ids)
    mech_records = []
    for m in mechanisms:
        tname = target_names.get(m.get("target_chembl_id"), None)
        if tname:
            mech_records.append({
                "molecule_chembl_id": m.get("molecule_chembl_id"),
                "target": tname,
                "action_type": m.get("action_type", ""),
                "mechanism_of_action": m.get("mechanism_of_action", ""),
            })
    smiles_map = fetch_smiles(drug_ids)
    targets = list(dict.fromkeys([r["target"] for r in mech_records]))
    return targets, mech_records, smiles_map

# ============== State ================
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
    target_protein: str = ""
    mechanism: str = ""
    admet_score: float = 0.0
    herg: float = 0.0
    bbb: float = 0.0
    cyp450: float = 0.0
    hepato_toxic: float = 0.0
    clinical_score: float = 0.0
    final_score: float = 0.0
    source: str = ""  # "ChEMBL 승인약물" or "AI 생성"

@dataclass
class PipelineState:
    disease: str
    targets: List[str] = field(default_factory=list)
    mechanisms: List[dict] = field(default_factory=list)
    target_description: str = ""
    data_source: str = ""
    hypothesis: str = ""
    candidates: List[Molecule] = field(default_factory=list)
    top_candidates: List[Molecule] = field(default_factory=list)
    reasoning: str = ""
    mechanism_summary: str = ""
    log: List[str] = field(default_factory=list)

def log(state: PipelineState, msg: str):
    state.log.append(msg)

# ============== Agents ================
def research_agent(state: PipelineState):
    log(state, f"[Research] ChEMBL에서 '{state.disease}' 승인약물 및 작용기전 실시간 조회 중...")
    targets, mechanisms, smiles_map = resolve_disease_pharmacology(state.disease)
    state.mechanisms = mechanisms
    state._smiles_map = smiles_map  # type: ignore

    if targets:
        state.targets = targets[:6]
        moas = list(dict.fromkeys([m["mechanism_of_action"] for m in mechanisms if m.get("mechanism_of_action")]))
        state.target_description = " / ".join(moas[:3]) if moas else f"{', '.join(state.targets)} 관련 기전"
        state.data_source = "ChEMBL (실시간 조회, 검증된 승인약물 기반)"
        log(state, f"[Research] ChEMBL에서 {len(state.targets)}개 실제 타겟 확인: {', '.join(state.targets)}")
    else:
        log(state, "[Research] ChEMBL에 매칭 데이터 없음 → PubMed + LLM으로 대체 조사")
        try:
            r = requests.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                              params={"db": "pubmed", "term": state.disease, "retmax": 3, "retmode": "json"},
                              timeout=5)
            ids = r.json()["esearchresult"]["idlist"]
            summ = requests.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
                                 params={"db": "pubmed", "id": ",".join(ids), "retmode": "json"}, timeout=5).json()
            literature = "\n".join([summ["result"][i]["title"] for i in ids if i in summ["result"]])
        except Exception:
            literature = ""
        target_raw = ask(
            f"질병: {state.disease}\n문헌: {literature}\n"
            "이 질병의 검증된 약물 표적(drug target) 단백질 3개를 쉼표로만 구분해 답해라. "
            "생물마커(biomarker)가 아닌 실제 약물이 결합하는 단백질만 답해라. 설명 없이 이름만."
        )
        state.targets = [t.strip() for t in target_raw.split(",") if t.strip()][:3]
        state.target_description = f"LLM 추론 기반 (ChEMBL 데이터 없음, 검증 필요)"
        state.data_source = "PubMed + LLM 추론 (⚠ 낮은 신뢰도, 전문가 검증 필요)"
    log(state, "[Research] 완료")

def hypothesis_agent(state: PipelineState):
    log(state, "[Hypothesis] 인과관계 기반 치료 가설 생성 중...")
    moa_lines = "\n".join([f"- {m['target']}: {m['mechanism_of_action']} ({m['action_type']})"
                            for m in state.mechanisms[:5] if m.get("mechanism_of_action")])
    prompt = (
        f"질병: {state.disease}\n"
        f"실제 검증된 약물 타겟: {', '.join(state.targets)}\n"
        f"ChEMBL 실제 작용기전 데이터:\n{moa_lines if moa_lines else state.target_description}\n\n"
        "위 데이터에 기반하여, '질병 원인 → 타겟 억제/활성화 → 신호전달 변화 → 증상 개선'의 "
        "명확한 인과관계 체인으로 치료 가설을 3문장 이내로 설명해라. "
        "생물마커를 타겟으로 착각하지 말고, 실제 결합 타겟만 언급해라."
    )
    state.hypothesis = ask(prompt)
    log(state, "[Hypothesis] 완료")

def mutate(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() < 2:
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
    try:
        f1 = Chem.GetMolFrags(m1, asMols=True)
        f2 = Chem.GetMolFrags(m2, asMols=True)
        combo = Chem.CombineMols(f1[0], f2[-1]) if f1 and f2 else m1
        Chem.SanitizeMol(combo)
        return Chem.MolToSmiles(combo)
    except Exception:
        return s1

def featurize(smiles: str, target: str = "", source: str = "AI 생성") -> Molecule:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Molecule(
        smiles=smiles, mol=mol,
        mw=Descriptors.MolWt(mol), logp=Crippen.MolLogP(mol),
        qed=QED.qed(mol), tpsa=Descriptors.TPSA(mol),
        hbd=Descriptors.NumHDonors(mol), hba=Descriptors.NumHAcceptors(mol),
        target_protein=target, source=source)

def generator_agent(state: PipelineState, n: int = 60):
    log(state, f"[Generator] ChEMBL 승인약물 구조 기반 {n}개 후보 생성 중...")
    smiles_map = getattr(state, "_smiles_map", {}) or {}
    mol_to_target = {m["molecule_chembl_id"]: m["target"] for m in state.mechanisms}

    seed_pairs = []
    for mid, smi in smiles_map.items():
        if Chem.MolFromSmiles(smi) is not None:
            seed_pairs.append((smi, mol_to_target.get(mid, "")))

    if not seed_pairs:
        log(state, "[Generator] ⚠ ChEMBL 승인약물 없음 → 범용 화학 骨格으로 대체")
        seed_pairs = [("CC(=O)Oc1ccccc1C(=O)O", ""), ("c1ccc2[nH]c3ccccc3c2c1", ""),
                      ("CCN(CC)CCNC(=O)c1ccc(N)cc1", "")]

    candidates = []
    for smi, tgt in seed_pairs:
        m = featurize(smi, target=tgt, source="ChEMBL 승인약물")
        if m:
            candidates.append(m)

    seed_smiles_only = [s for s, _ in seed_pairs]
    attempts = 0
    while len(candidates) < n and attempts < n * 5:
        attempts += 1
        a, b = np.random.choice(seed_smiles_only, 2, replace=True)
        s = mutate(a) if np.random.rand() < 0.5 else crossover(a, b)
        if Chem.MolFromSmiles(s) is not None:
            src_target = ""
            for seed_s, seed_t in seed_pairs:
                if seed_s == a:
                    src_target = seed_t
                    break
            m = featurize(s, target=src_target, source="AI 변형 (ChEMBL 유도체)")
            if m:
                candidates.append(m)

    unique = list({c.smiles: c for c in candidates}.values())
    state.candidates = unique[:n]
    log(state, f"[Generator] {len(state.candidates)}개 유효 후보 생성 (중복 제거, ChEMBL 실제 약물 {len(seed_pairs)}개 포함)")

def molecule_to_mps_entropy(mol: Molecule, bond_dim: int = 8) -> float:
    try:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol.mol, 2, nBits=64)
        bits = np.array(fp, dtype=float).reshape(8, 8) + 1e-9
        mps = qtn.MatrixProductState.from_dense(
            (bits.reshape(-1) / np.linalg.norm(bits.reshape(-1))), dims=[2] * 6, max_bond=bond_dim)
        ent = sum([mps.entropy(i) for i in range(1, min(mps.L, 6))]) / max(1, min(mps.L - 1, 5))
        return float(ent)
    except Exception:
        return 0.1

def quantum_agent(state: PipelineState, top_k: int = 20):
    log(state, "[Quantum] Tensor Network(MPS) 탐색 공간 압축 중...")
    for m in state.candidates:
        m.bond_entropy = molecule_to_mps_entropy(m)
    ranked = sorted(state.candidates, key=lambda m: -(m.qed * 0.6 + m.bond_entropy * 0.4))
    state.candidates = ranked[:top_k]
    log(state, f"[Quantum] Top-{top_k} 후보 선정 완료")

def docking_agent(state: PipelineState):
    log(state, "[Docking] 타겟별 결합친화도(heuristic) 계산 중...")
    for m in state.candidates:
        if not m.target_protein and state.targets:
            m.target_protein = np.random.choice(state.targets)
        score = -(m.qed * 4 + (1 / (1 + abs(m.logp - 2.5))) * 3 + (150 - min(m.tpsa, 150)) / 50)
        m.binding_score = round(score, 3)
        moa = next((mm["mechanism_of_action"] for mm in state.mechanisms
                    if mm["target"] == m.target_protein and mm.get("mechanism_of_action")), None)
        m.mechanism = moa if moa else f"{m.target_protein} 조절 추정"
    state.candidates.sort(key=lambda m: m.binding_score)
    log(state, "[Docking] 완료")

def admet_agent(state: PipelineState):
    log(state, "[ADMET] 약물성 상세 평가 중...")
    for m in state.candidates:
        lipinski = sum([m.mw <= 500, m.logp <= 5, m.hbd <= 5, m.hba <= 10])
        m.herg = round(1.0 - min(abs(m.logp - 2.0) / 5, 1), 3)
        m.bbb = round(1.0 if m.tpsa < 60 else 0.5, 3)
        m.cyp450 = round(max(0, 1 - m.mw / 500), 3)
        m.hepato_toxic = round(1.0 if m.qed > 0.7 else 0.5, 3)
        m.admet_score = round((lipinski / 4) * 0.4 + m.qed * 0.3 + m.herg * 0.1 + m.hepato_toxic * 0.2, 3)
    log(state, "[ADMET] 완료")

def clinical_agent(state: PipelineState):
    log(state, "[Clinical] 임상 가능성 종합 평가 중...")
    for m in state.candidates:
        source_bonus = 0.05 if m.source == "ChEMBL 승인약물" else 0.0
        m.clinical_score = round(m.qed * 0.5 + m.admet_score * 0.5, 3)
        m.final_score = round(m.clinical_score * 0.4 + m.admet_score * 0.3 +
                               (-m.binding_score) * 0.03 + source_bonus, 3)
    state.candidates.sort(key=lambda m: -m.final_score)
    state.top_candidates = state.candidates[:10]
    log(state, "[Clinical] 완료")

def reasoning_agent(state: PipelineState):
    log(state, "[Reasoning] 최종 근거 생성 중...")
    top = state.top_candidates[0]
    prompt = (
        f"질병: {state.disease}\n"
        f"타겟: {top.target_protein}\n"
        f"치료 가설: {state.hypothesis}\n"
        f"선정 분자: {top.smiles} (출처: {top.source})\n"
        f"기전: {top.mechanism}\n"
        f"QED={top.qed:.3f} Binding={top.binding_score:.2f} ADMET={top.admet_score:.3f}\n\n"
        f"'{top.target_protein}'을 조절하여 '{state.disease}'가 개선되는 인과관계를 "
        "질병원인→타겟→분자작용→임상효과 순서로 4문장 이내로 설명해라."
    )
    state.reasoning = ask(prompt)
    state.mechanism_summary = (
        f"[인과관계 흐름]\n"
        f"1. 질병: {state.disease}\n"
        f"2. 검증된 타겟: {top.target_protein} (데이터 출처: {state.data_source})\n"
        f"3. 분자 작용: {top.mechanism}\n"
        f"4. 분자 출처: {top.source}\n\n"
        f"SMILES: {top.smiles}\n"
        f"Binding={top.binding_score:.3f} | ADMET={top.admet_score:.3f} | Final={top.final_score:.3f}"
    )
    log(state, "[Reasoning] 완료")

def report_agent(state: PipelineState) -> str:
    lines = [
        f"# 신약개발 보고서: {state.disease}",
        "",
        f"## 데이터 출처\n{state.data_source}",
        "",
        "## 검증된 약물 타겟",
        f"{', '.join(state.targets) if state.targets else '없음'}",
        f"{state.target_description}",
        "",
        "## 치료 가설", state.hypothesis, "",
    ]
    if state.top_candidates:
        top = state.top_candidates[0]
        lines += [
            "## 최우선 후보",
            f"SMILES: `{top.smiles}` (출처: {top.source})",
            f"타겟: {top.target_protein} | 기전: {top.mechanism}",
            f"MW={top.mw:.1f} LogP={top.logp:.2f} QED={top.qed:.3f} Final={top.final_score:.3f}",
            "", "## 근거", state.reasoning, "",
        ]
    lines += ["## Top 10 후보", ""]
    for i, m in enumerate(state.top_candidates, 1):
        lines.append(f"{i}. `{m.smiles}` | {m.target_protein} | {m.source} | Final={m.final_score:.3f}")
    return "\n".join(lines)

def run_pipeline(disease: str, n_candidates: int, top_k: int, status_box):
    state = PipelineState(disease=disease)
    steps = [
        ("Research", research_agent, {}), ("Hypothesis", hypothesis_agent, {}),
        ("Generator", generator_agent, {"n": n_candidates}),
        ("Quantum", quantum_agent, {"top_k": top_k}),
        ("Docking", docking_agent, {}), ("ADMET", admet_agent, {}),
        ("Clinical", clinical_agent, {}), ("Reasoning", reasoning_agent, {}),
    ]
    for name, fn, kwargs in steps:
        status_box.write(f"▶ {name} Agent 실행 중...")
        fn(state, **kwargs)
        status_box.write(f"✔ {name} Agent 완료")
    return state

# ============== UI ================
st.set_page_config(page_title="Quantum-Agent Drug Discovery", layout="wide")
st.title("Quantum-Agent Drug Discovery Platform")
st.caption("Multi-Agent AI Pipeline for Target Identification, Molecular Generation, and ADMET Profiling")

with st.sidebar:
    st.header("Pipeline Configuration")
    disease = st.text_input("Disease Name", "Heart Failure")
    n_candidates = st.slider("Candidate Molecules", 20, 200, 60)
    top_k = st.slider("MPS Compression Top-K", 5, 50, 20)
    run = st.button("Run Pipeline", type="primary")

if run and disease:
    status_box = st.empty()
    log_container = st.container()
    with st.spinner("Running pipeline..."):
        state = run_pipeline(disease, n_candidates, top_k, log_container)
    st.session_state["state"] = state
    st.session_state.pop("messages", None)
    st.success("Pipeline completed")

if "state" in st.session_state:
    state = st.session_state["state"]
    tabs = st.tabs(["Summary", "Scientific Rationale", "MPS Visualization",
                     "Candidate Molecules", "Ask Ollama", "Report"])

    with tabs[0]:
        st.subheader("Data Source")
        if "ChEMBL" in state.data_source and "⚠" not in state.data_source:
            st.success(state.data_source)
        else:
            st.warning(state.data_source)

        st.subheader("Validated Drug Targets")
        st.write(", ".join(state.targets) if state.targets else "None identified")
        st.caption(state.target_description)

        st.subheader("Therapeutic Hypothesis")
        st.info(state.hypothesis)

        if state.top_candidates:
            top = state.top_candidates[0]
            st.subheader("Lead Candidate")
            st.code(top.smiles)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Molecular Weight", f"{top.mw:.1f}")
            c2.metric("QED", f"{top.qed:.3f}")
            c3.metric("Final Score", f"{top.final_score:.3f}")
            c4.metric("Source", top.source)
            st.caption(f"Target: {top.target_protein} | Mechanism: {top.mechanism}")

        st.subheader("Execution Log")
        st.code("\n".join(state.log))

    with tabs[1]:
        if state.top_candidates:
            st.markdown(state.mechanism_summary)
            st.subheader("Rationale")
            st.write(state.reasoning)

    with tabs[2]:
        entropies = [m.bond_entropy for m in state.candidates]
        scores = [m.final_score for m in state.candidates]
        fig = go.Figure(data=go.Scatter(
            x=entropies, y=scores, mode="markers",
            marker=dict(size=10, color=scores, colorscale="Viridis", showscale=True),
            text=[m.target_protein for m in state.candidates],
            hovertemplate="<b>%{text}</b><br>Entropy: %{x:.3f}<br>Score: %{y:.3f}<extra></extra>"
        ))
        fig.update_layout(xaxis_title="Bond Entanglement Entropy", yaxis_title="Final Score",
                           title="MPS Search Space Compression", height=600)
        st.plotly_chart(fig, use_container_width=True)

    with tabs[3]:
        for i, m in enumerate(state.top_candidates, 1):
            with st.expander(f"#{i}  {m.smiles[:35]}...  |  {m.target_protein}  |  {m.source}"):
                st.code(m.smiles)
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("MW", f"{m.mw:.1f}")
                c2.metric("LogP", f"{m.logp:.2f}")
                c3.metric("QED", f"{m.qed:.3f}")
                c4.metric("Binding", f"{m.binding_score:.2f}")
                c5.metric("ADMET", f"{m.admet_score:.3f}")
                st.write(f"Mechanism: {m.mechanism}")
                st.write(f"hERG {m.herg:.3f}  |  BBB {m.bbb:.3f}  |  CYP450 {m.cyp450:.3f}  |  Hepatotoxicity {m.hepato_toxic:.3f}")

    with tabs[4]:
        st.subheader("Ask Ollama")
        st.caption("Discuss the pipeline results with the language model")
        if "messages" not in st.session_state:
            st.session_state.messages = [("assistant", f"Ask me anything about the {state.disease} results.")]
        for role, msg in st.session_state.messages:
            with st.chat_message(role):
                st.write(msg)
        user_input = st.chat_input("Type your question...")
        if user_input:
            st.session_state.messages.append(("user", user_input))
            with st.chat_message("user"):
                st.write(user_input)
            top = state.top_candidates[0] if state.top_candidates else None
            context = (
                f"질병: {state.disease}\n타겟: {', '.join(state.targets)}\n"
                f"가설: {state.hypothesis}\n"
                f"최우선 후보: {top.smiles if top else 'N/A'} (타겟:{top.target_protein if top else ''})\n"
                f"질문: {user_input}"
            )
            response = ask(context)
            st.session_state.messages.append(("assistant", response))
            with st.chat_message("assistant"):
                st.write(response)

    with tabs[5]:
        report = report_agent(state)
        st.markdown(report)
        st.download_button("Download Report", report, file_name=f"{state.disease}_report.md")
