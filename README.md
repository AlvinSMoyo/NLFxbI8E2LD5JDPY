# Beyond Keywords: Building an AI-Powered Talent Sourcing Pipeline

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![NLP](https://img.shields.io/badge/NLP-Semantic_Search-orange)

This repository contains the full implementation of a machine learning pipeline for intelligent candidate sourcing. The project explores how NLP, semantic embeddings, and supervised learning can be combined to enhance traditional recruitment workflows.

> ⚠️ Repository Name: `NLFxbI8E2LD5JDPY` — named as per Apziva’s submission protocol.

---

## 🚀 Project Objectives

- Predict candidate-role fit using available metadata
- Rank candidates based on keyword and semantic similarity
- Adapt to recruiter feedback by re-ranking candidates
- Cluster candidates to surface talent segments
- Simulate suitability labels and train a predictive classifier

---

## 🛠️ Tech Stack

- **Language**: Python 3.x  
- **Libraries**: `pandas`, `scikit-learn`, `nltk`, `sentence-transformers`, `wordcloud`, `umap-learn`, `hdbscan`, `matplotlib`, `seaborn`
- **Embedding Models**:
  - Sentence-BERT
  - Word2Vec (Gensim)
  - GloVe
  - FastText
- **Clustering Techniques**:
  - KMeans (TF-IDF-based)
  - UMAP + HDBSCAN (S-BERT based)

---

## 📁 Repository Structure

---

## 📊 Key Modules

### 🔹 TF-IDF Similarity Ranking
- Measures lexical overlap with target role descriptions

### 🔹 S-BERT Semantic Ranking
- Captures deeper semantic alignment of candidate profiles
- Re-ranking simulation based on starred candidate

### 🔹 Word2Vec / GloVe / FastText Benchmarking
- Compared embeddings to assess sensitivity to phrasing and role structure

### 🔹 Clustering & Profiling
- KMeans (TF-IDF) for transparent segmentation
- UMAP + HDBSCAN (exploratory deep clustering)

### 🔹 Suitability Classifier
- Simulated fit labels (e.g., High/Medium/Low)
- Trained Random Forest model to score candidate fit

---

## 📷 Visuals

![Re-ranking Dumbbell Chart](./figures/dumbbell_chart.png)  
*Rank shifts before vs after recruiter feedback*

![Job Title WordCloud](./figures/wordcloud.png)  
*Most frequent role-related terms*

![TF-IDF Clustering](./figures/cluster_map.png)  
*Clustered candidate segments based on TF-IDF*

---

## 🧪 Getting Started

```bash
# 1. Clone the repo
git clone https://github.com/AlvinSMoyo/NLFxbI8E2LD5JDPY.git

# 2. Install dependencies
pip install -r requirements.txt

# 3. Open the notebook
jupyter notebook notebook/Potential_Talents_AI.ipynb


