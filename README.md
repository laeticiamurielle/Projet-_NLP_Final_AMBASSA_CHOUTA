le  fichier `README.md`, structuré de manière professionnelle et adapté au flux de travail réel de votre notebook `projet_final.ipynb` (incluant l'analyse de similarité et la corrélation budgétaire).
# 📊 Audit SND30 — Intelligence Artificielle et Finances Publiques

**ISE3-DS · ISSEA · Yaoundé — Année académique 2025-2026**

> *Comment l'IA peut-elle mesurer mathématiquement l'évolution des priorités de l'État camerounais entre la Loi de Finances 2024 et les perspectives de 2025-2026 ? Existe-t-il un alignement statistiquement significatif entre le discours budgétaire et les piliers de la SND30 ?*
## 📋 Sommaire

1. [Architecture du Projet]
2. [Pipeline de Traitement]
3. [Méthodologie NLP]
4. [Analyse de Corrélation]
5. [Installation]

---# 📊 Analyse NLP Finance - Classification SND30

Ce projet utilise des techniques de **Natural Language Processing (NLP)** pour analyser, nettoyer et segmenter les lois de finances (2024-2025) et les aligner avec les piliers stratégiques de la **SND30** (Stratégie Nationale de Développement).

## 🚀 Fonctionnalités
- **Extraction & Nettoyage** : Traitement automatique des textes bruts des lois de finances.
- **Classification Zero-Shot** : Utilisation du modèle `facebook/bart-large-mnli` pour classer les libellés budgétaires sans entraînement préalable.
- **Segmentation** : Identification des montants (AE/CP) par pilier stratégique.
- **Visualisation** : Analyse de l'effort financier versus la visibilité sémantique.

## 🛠️ Installation & Configuration

Ce projet utilise **Poetry** pour une gestion rigoureuse des dépendances et de l'environnement virtuel.

### 1. Prérequis
- Python 3.11 ou 3.12
- Poetry installé sur votre système

### 2. Initialisation de l'environnement
Dans le terminal de VS Code, à la racine du projet :

```bash
# Configurer Poetry pour créer le venv dans le dossier du projet
poetry config virtualenvs.in-project true

# Installer les dépendances (torch, transformers, pandas, etc.)
poetry install



## Installation

### Prérequis
- Python 3.10 ou 3.11
- [Poetry](https://python-poetry.org/docs/#installation)

```bash
# Cloner le dépôt
git clone https://github.com/votre-groupe/audit-snd30.git
cd audit-snd30

# Installer les dépendances avec Poetry
poetry install

# Activer l'environnement virtuel
poetry shell
```

---

## Utilisation

### 1. Pipeline complet (extraction → classification → analyse)

```bash
# Avec les PDFs du MINFI
poetry run python scripts/run_pipeline.py \
    --"C:\Users\Laeticia\Desktop\Loi_Finance\LOI DES FINANCES 2023-2024.pdf" \
    --"C:\Users\Laeticia\Desktop\Loi_Finance\LOI DES FINANCES 2024-2025.pdf"

# Si l'extraction a déjà été faite (données dans data/processed/)
poetry run python scripts/run_pipeline.py --skip-extraction

# Zero-shot uniquement (sans fine-tuning, plus rapide)
poetry run python scripts/run_pipeline.py --skip-extraction --skip-finetuning
```

### 2. Extraction seule (CLI)

```bash
# Commande enregistrée par Poetry
snd30-extract --lf 2024 --pdf /chemin/LF_2023-2024.pdf
snd30-extract --lf 2025 --pdf /chemin/LF_2024-2025.pdf
```

### 3. Dashboard interactif

```bash
poetry run streamlit run audit_snd30/dashboard/app.py
```
→ Ouvrir http://localhost:8501

### 4. Tests unitaires

```bash
poetry run pytest tests/ -v

# Avec rapport de couverture
poetry run pytest tests/ -v --cov=audit_snd30 --cov-report=html
```

---

## Pipeline NLP — Détail méthodologique

```
PDF MINFI
   │
   ▼ Étape 1 — Extraction (pdfplumber)
┌──────────────────────────────────────┐
│  Détection des pages budgétaires     │
│  Extraction CODE · LIBELLE · AE · CP │
│  Corrections artefacts OCR           │
└──────────────────────────────────────┘
   │
   ▼ Étape 2 — Classification NLP (CamemBERT)
┌──────────────────────────────────────┐
│  2a. Zero-shot (XLM-RoBERTa-XNLI)   │
│      → Pseudo-labels sans annotation │
│  2b. Fine-tuning (camembert-base)    │
│      → Spécialisation vocabulaire    │
│         budgétaire camerounais       │
│  2c. Prédiction finale               │
│      → PILIER_SND30 · CONFIANCE      │
└──────────────────────────────────────┘
   │
   ▼ Étape 3 — Glissement Sémantique
┌──────────────────────────────────────┐
│  Jensen-Shannon (distributions)      │
│  TF-IDF Cosinus (vocabulaire)        │
│  Δ Part AE/CP (ressources)           │
└──────────────────────────────────────┘
   │
   ▼ Étape 4 — Alignement Statistique
┌──────────────────────────────────────┐
│  Test du Chi² vs cibles SND30        │
│  H₀ : budget aligné (p > 0.05)       │
│  H₁ : désaligné (p < 0.05)           │
└──────────────────────────────────────┘
```




### 2. Modèle d'Embedding

Nous utilisons `transformers` avec le modèle `camembert-base` pour générer des représentations vectorielles :

```python
# Extrait du notebook
from transformers import AutoTokenizer, AutoModel
# Génère un vecteur de 768 dimensions pour chaque article
embedding = generer_un_embedding(texte_nettoye)

```

## 📈 Analyse de Corrélation (SND30 vs Budget)

Une partie cruciale du notebook est le calcul du **Bilan des Piliers**.

### Indicateurs clés :

* **Frequence_Thematique** : Nombre de fois qu'un pilier de la SND30 est détecté dans les textes.
* **AE_Mrd** : Autorisations d'Engagement (en milliards de FCFA) allouées.

### Interprétation du Regplot :

* **Alignement Parfait** : Les points suivent la ligne de tendance. Le discours politique correspond aux investissements.
* **Écart (Gap)** : Un pilier avec une forte fréquence mais un faible budget indique un "effet d'annonce".
* **Priorité Financière** : Un pilier avec un fort budget mais une faible fréquence indique une gestion technique peu détaillée dans les textes de loi.

## 🚀 Usage

### Dans le Notebook

Exécutez les cellules dans l'ordre pour :

1. Charger les PDF des lois de finances.
2. Générer les matrices de similarité cosinus (Moyenne observée : `~0.97` entre 2024 et 2025).
3. Visualiser le nuage de points "Arbitrage Budgétaire".

### Variables Requises

Le DataFrame final `bilan_piliers` doit contenir :

* `Pilier_Moteur` : Transformation structurelle, Capital humain, etc.
* `Frequence_Thematique` : Score d'occurrence.
* `AE_Mrd` : Valeur financière.

## 🛠 Dépendances

* `torch` & `transformers` (CamemBERT)
* `spacy` (Modèle `fr_core_news_md`)
* `pdfplumber`
* `pandas` & `seaborn` (Visualisation)

---

*Ce projet a été développé dans le cadre de l'analyse NLP des finances publiques .*
