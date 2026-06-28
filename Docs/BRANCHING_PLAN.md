# Tentative Branch Plan

This file documents the recommended branch structure for the CrediRisk RAG team workflow.

## Branch Structure

```text
main
└── dev
    ├── feature/data-exploration
    ├── feature/modeling-lightgbm
    ├── feature/nlp-distilbert
    ├── feature/rag-retrieval
    ├── feature/letter-generation
    ├── feature/evaluation
    └── docs/report-and-readme
```

## Branch Purpose

- `main`: Stable final project branch. Only merge tested, complete work here.
- `dev`: Integration branch for combining team work before promoting to `main`.
- `feature/data-exploration`: Lending Club data loading, cleaning checks, EDA notebooks, target definition, and class balance analysis.
- `feature/modeling-lightgbm`: Tabular loan default modeling with LightGBM.
- `feature/nlp-distilbert`: Borrower text encoding and DistilBERT experiments.
- `feature/rag-retrieval`: Regulatory document collection, chunking, embeddings, and retrieval.
- `feature/letter-generation`: RAG/LangGraph pipeline for adverse-action letter generation.
- `feature/evaluation`: SHAP validation, model metrics, calibration, and letter quality rubric.
- `docs/report-and-readme`: README, report, proposal files, diagrams, and documentation.

## Initial Git Setup

Clone the repository:

```bash
git clone git@github.com:lindaperez/CreditRiskRAG.git
cd CreditRiskRAG
```

If SSH is not configured, use HTTPS:

```bash
git clone https://github.com/lindaperez/CreditRiskRAG.git
cd CreditRiskRAG
```

## Create Branches

Start from the latest `main`:

```bash
git checkout main
git pull origin main
```

Create and push `dev`:

```bash
git checkout -b dev
git push -u origin dev
```

Create feature branches from `dev`:

```bash
git checkout dev
git checkout -b feature/data-exploration
git push -u origin feature/data-exploration

git checkout dev
git checkout -b feature/modeling-lightgbm
git push -u origin feature/modeling-lightgbm

git checkout dev
git checkout -b feature/nlp-distilbert
git push -u origin feature/nlp-distilbert

git checkout dev
git checkout -b feature/rag-retrieval
git push -u origin feature/rag-retrieval

git checkout dev
git checkout -b feature/letter-generation
git push -u origin feature/letter-generation

git checkout dev
git checkout -b feature/evaluation
git push -u origin feature/evaluation

git checkout dev
git checkout -b docs/report-and-readme
git push -u origin docs/report-and-readme
```

Return to `dev` after creating branches:

```bash
git checkout dev
```

## Daily Workflow

Before starting work:

```bash
git checkout dev
git pull origin dev
git checkout feature/your-branch-name
git merge dev
```

After making changes:

```bash
git status
git add .
git commit -m "Short description of the change"
git push
```

Then open a pull request from the feature branch into `dev`.

## Merge Workflow

Recommended flow:

```text
feature branch -> dev -> main
```

Use pull requests for all merges into `dev` and `main`.
