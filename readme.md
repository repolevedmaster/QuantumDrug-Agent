# 🧬 Quantum-Agent Drug Discovery Platform

> Multi-Agent AI 기반 차세대 신약 후보 탐색 플랫폼
>
> **LLM + Tensor Network(MPS) + Molecular Generation + Docking + ADMET + Clinical Ranking**

---

# Overview

Quantum-Agent Drug Discovery Platform은 질병 정보를 입력하면 AI가

- 최신 논문 조사
- 표적 단백질 추론
- 치료 가설 생성
- 후보 분자 생성
- Tensor Network(MPS) 기반 탐색 공간 압축
- Docking
- ADMET 평가
- 임상 성공 가능성 평가
- 최종 후보 선정 이유 생성

까지 자동으로 수행하는 Multi-Agent 신약 탐색 시스템입니다.

모든 과정은 하나의 Pipeline으로 연결되어 있으며 Streamlit 기반 GUI에서 실행됩니다.

---

# Features

## Research Agent

- PubMed 논문 검색
- 최신 연구 제목 수집
- 핵심 표적 단백질 자동 추론
- RCSB PDB 구조 검색
- Receptor 자동 확보

---

## Hypothesis Agent

질병과 논문 정보를 기반으로

- 치료 전략 생성
- 작용기전 생성
- LLM 기반 치료 가설 생성

---

## Generator Agent

후보 약물을 생성합니다.

기능

- Seed Molecule
- Mutation
- Crossover
- RDKit 기반 분자 생성

---

## Quantum Agent (Tensor Network)

이 프로젝트의 핵심 기술입니다.

분자를

Fingerprint

↓

Tensor

↓

Matrix Product State(MPS)

로 변환하여

각 분자의

- Bond Entanglement
- Entropy

를 계산합니다.

이를 이용하여

수백 개의 후보 중

가장 정보량이 높은 후보만 남겨

탐색 공간을 크게 줄입니다.

---

## Docking Agent

두 가지 모드를 지원합니다.

### AutoDock Vina

실제 Docking 수행

또는

### Heuristic Docking

OpenBabel 또는 Vina가 없을 경우

RDKit Descriptor 기반 점수 계산

---

## ADMET Agent

두 가지 평가 방식을 지원합니다.

### ChemBERTa

사전학습 분자 언어모델 이용

또는

### RDKit Descriptor

Lipinski Rule

QED

등을 이용한 평가

---

## Clinical Agent

최종 후보를 선정합니다.

평가 요소

- Binding Score
- ADMET
- Drug-likeness
- Clinical Success Score

---

## Reasoning Agent

LLM이

왜 이 후보가 가장 적합한지

과학적인 설명을 생성합니다.

---

# Architecture

```
Disease
   │
   ▼
Research Agent
   │
   ▼
Hypothesis Agent
   │
   ▼
Generator Agent
   │
   ▼
Quantum MPS Agent
   │
   ▼
Docking Agent
   │
   ▼
ADMET Agent
   │
   ▼
Clinical Agent
   │
   ▼
Reasoning Agent
   │
   ▼
Markdown Report
```

---

# Technology Stack

## AI

- Ollama
- Llama 3.1
- LangChain

---

## Molecular AI

- RDKit
- ChemBERTa

---

## Quantum-inspired

- Quimb
- Matrix Product State (MPS)
- Tensor Network

---

## Drug Discovery

- AutoDock Vina
- OpenBabel
- PubMed API
- RCSB PDB

---

## Visualization

- Streamlit
- Plotly

---

# Installation

```bash
git clone https://github.com/your-id/Quantum-Agent-DrugDiscovery.git

cd Quantum-Agent-DrugDiscovery
```

설치

```bash
pip install -r requirements.txt
```

Ollama 모델 다운로드

```bash
ollama pull llama3.1
```

실행

```bash
streamlit run app.py
```

---

# Example Pipeline

```
Disease

↓

Research

↓

Target Protein

↓

Hypothesis

↓

Candidate Molecules

↓

Tensor Network Compression

↓

Docking

↓

ADMET

↓

Clinical Ranking

↓

Top Drug Candidate

↓

Scientific Reasoning

↓

Markdown Report
```

---

# Screenshots

추후 추가 예정

- Dashboard
- MPS Visualization
- Candidate Ranking
- Report

---

# Future Work

- Graph Neural Network 기반 분자 생성
- Diffusion Model 분자 생성
- Reinforcement Learning 기반 최적화
- AlphaFold 구조 활용
- Multi-target Drug Discovery
- Protein Language Model 적용
- 실제 Binding Site 자동 탐색
- GPU Tensor Network 최적화

---

# Project Structure

```
.
├── app.py
├── requirements.txt
├── README.md
└── report.md
```

---

# License

MIT License

---

# Disclaimer

본 프로젝트는 연구 및 교육 목적으로 개발되었습니다.

생성된 후보 물질은 실제 의약품이 아니며, 실제 임상 또는 의료 목적으로 사용할 수 없습니다.

---

# Author

Quantum-Agent Drug Discovery Platform

Developed with

- Python
- Streamlit
- LangChain
- RDKit
- Quimb
- ChemBERTa
- AutoDock Vina
- Ollama