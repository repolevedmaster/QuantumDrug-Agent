# 🧬 Quantum-Agent Drug Discovery Platform

> Next-generation AI-powered drug discovery platform based on Multi-Agent systems
>
> **LLM + Tensor Networks (MPS) + Molecular Generation + Docking + ADMET + Clinical Ranking**

---

# Overview

The Quantum-Agent Drug Discovery Platform is an automated AI system that takes a disease as input and performs:

- Scientific literature retrieval
- Target protein inference
- Therapeutic hypothesis generation
- Molecular candidate generation
- Tensor Network (MPS)-based search space compression
- Molecular docking simulation
- ADMET evaluation
- Clinical success prediction
- Final candidate ranking and reasoning generation

All processes are connected in a unified pipeline and executed through a Streamlit-based GUI.

---

# Features

## Research Agent

- PubMed literature search
- Extraction of recent research papers
- Automatic target protein inference
- RCSB PDB structure retrieval
- Automatic receptor acquisition

---

## Hypothesis Agent

Based on disease context and literature:

- Therapeutic strategy generation
- Mechanism of action inference
- LLM-based hypothesis generation

---

## Generator Agent

Generates molecular candidates using:

- Seed molecules
- Mutation operations
- Crossover operations
- RDKit-based molecular generation

---

## Quantum Agent (Tensor Network)

This is the core innovation of the system.

Molecules are transformed into:

Fingerprint → Tensor → Matrix Product State (MPS)

This representation is used to compute:

- Bond entanglement
- Entropy

Using this, the system reduces a large molecular search space by selecting only the most informative candidates.

---

## Docking Agent

Supports two modes:

### AutoDock Vina

Performs real molecular docking simulation

### Heuristic Docking

Fallback mode used when Vina/OpenBabel is unavailable:

- RDKit descriptor-based scoring

---

## ADMET Agent

Two evaluation modes:

### ChemBERTa

Uses a pretrained molecular language model

### RDKit Descriptor-Based Evaluation

- Lipinski Rule of Five
- QED (drug-likeness score)
- Physicochemical descriptors

---

## Clinical Agent

Ranks final candidates using:

- Binding score
- ADMET score
- Drug-likeness
- Clinical success prediction score

---

## Reasoning Agent

Uses an LLM to generate scientific explanations for:

- Why the selected molecule is optimal
- How it aligns with the therapeutic hypothesis
- Mechanistic justification

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

## Quantum-Inspired Computing

- Quimb
- Matrix Product State (MPS)
- Tensor Networks

---

## Drug Discovery Tools

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
git clone https://github.com/repolevedmaster/QuantumDrug-Agent

cd Quantum-Agent-DrugDiscovery
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Download Ollama model:

```bash
ollama pull llama3.1
```

Run the application:

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

To be added later

- Dashboard
- MPS Visualization
- Candidate Ranking
- Report View

---

# Future Work

- Graph Neural Networks for molecule generation
- Diffusion models for molecular design
- Reinforcement learning optimization
- AlphaFold structure integration
- Multi-target drug discovery
- Protein language model integration
- Automated binding site detection
- GPU-accelerated tensor networks

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

This project is developed for research and educational purposes only.

The generated compounds are not actual drugs and must not be used for clinical or medical purposes.

---

# Author

Quantum-Agent Drug Discovery Platform

Developed with:

- Python
- Streamlit
- LangChain
- RDKit
- Quimb
- ChemBERTa
- AutoDock Vina
- Ollama
