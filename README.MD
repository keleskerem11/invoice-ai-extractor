# 📄 Invoice AI Extractor

> Extraction automatique de données de factures PDF via IA générative — Mode single ou batch consolidé.

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Mistral AI](https://img.shields.io/badge/Mistral_AI-Pixtral-FF7000.svg)](https://mistral.ai)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🎯 Le problème résolu

Les comptables, gestionnaires et dirigeants de PME passent **plusieurs heures par jour** à saisir manuellement les données de factures fournisseurs dans leurs outils comptables.

**Invoice AI Extractor** automatise cette tâche : uploadez une (ou plusieurs) facture(s) PDF, l'IA extrait toutes les données structurées en quelques secondes, exportables en JSON, Excel ou CSV.

---

## ✨ Fonctionnalités

### Mode Single — Une facture en détail
- Upload PDF (toutes mises en page)
- Extraction IA via **Mistral Pixtral** (multimodal, supporte le multi-pages)
- Affichage structuré : fournisseur, client, lignes, TVA, acomptes
- Export en **JSON / Excel multi-feuilles / CSV**

### Mode Batch — Plusieurs factures consolidées
- Upload simultané de **plusieurs PDF**
- Traitement séquentiel avec **barre de progression**
- Gestion d'erreurs individuelle (un échec ne bloque pas les autres)
- Rapport consolidé : tableau récapitulatif + statistiques globales
- Export Excel **3 feuilles** : Récapitulatif / Erreurs / Statistiques

### Données extraites
- Numéro de facture, dates (émission, échéance, livraison)
- Fournisseur (nom, adresse, SIRET, TVA, IBAN, BIC, contact)
- Client (nom, contact, adresse, identifiants fiscaux)
- Lignes de facture détaillées (référence, désignation, quantité, prix, remises, taux TVA)
- **Ventilation TVA multi-taux** (5,5% / 10% / 20%)
- **Acomptes versés** avec dates et descriptions
- Totaux complets (HT, TVA, TTC, net à payer)
- Mode et conditions de paiement, devise

---

## 🚀 Démo

**Démo en ligne** : *à venir (Streamlit Cloud)*

**Captures d'écran** : *à venir*

---

## 🛠️ Stack technique

| Couche | Outil |
|---|---|
| **Langage** | Python 3.13 |
| **Framework UI** | Streamlit |
| **IA / LLM** | Mistral AI — `pixtral-12b-latest` (multimodal vision) |
| **Manipulation PDF** | PyMuPDF (`fitz`) |
| **Data wrangling** | pandas |
| **Export Excel** | openpyxl |
| **Gestion secrets** | python-dotenv |

---

## 📦 Installation locale

### Prérequis
- Python 3.10+
- Une clé API Mistral AI ([création gratuite ici](https://console.mistral.ai))

### Étapes

```bash
# 1. Cloner le repo
git clone https://github.com/keleskerem11/invoice-ai-extractor.git
cd invoice-ai-extractor

# 2. Créer et activer un environnement virtuel
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Créer un fichier .env à la racine avec ta clé Mistral
echo MISTRAL_API_KEY=ta_cle_ici > .env

# 5. Lancer l'app
streamlit run app.py
```

L'app s'ouvre automatiquement dans ton navigateur sur `http://localhost:8501`.

---

## 🧠 Architecture
invoice-ai-extractor/
├── app.py                 # Application Streamlit (UI + logique)
├── Facture/               # Factures PDF de test
├── .env                   # Clé API (non versionné)
├── .gitignore
├── requirements.txt       # Dépendances Python
└── README.md

### Flux d'extraction
PDF uploadé
↓
Conversion en images PNG (PyMuPDF, 150 DPI)
↓
Encodage base64
↓
Envoi à l'API Mistral Pixtral avec prompt structuré (JSON schema strict)
↓
Parsing JSON → validation des champs
↓
Affichage UI + Export multi-format

### Prompt engineering

Le prompt utilisé applique plusieurs bonnes pratiques :
- **Schéma JSON exhaustif** avec types explicites
- **Règles strictes numérotées** (10 règles dont gestion des acomptes, multi-TVA, devises)
- **Self-check final** : l'IA vérifie sa réponse avant de la retourner
- **Mode JSON forcé** côté API (`response_format={"type": "json_object"}`)

---

## 🗺️ Roadmap

- [ ] Déploiement Streamlit Cloud + URL publique
- [ ] Captures d'écran et GIF de démo dans le README
- [ ] Visualisation PDF côte à côte avec extraction
- [ ] Export direct vers logiciels comptables (Pennylane, Sage, EBP)
- [ ] Mode "correction inline" : éditer les champs extraits avant export
- [ ] Détection automatique des doublons sur traitements batch
- [ ] Authentification utilisateur + historique des extractions

---

## 👤 Auteur

**Kerem Keles**
Data Analyst freelance — Python · SQL · IA générative
Le Havre, France

- Profil Malt : *à venir*
- GitHub : [@keleskerem11](https://github.com/keleskerem11)
- Contact : keles.kerem11@gmail.com

---

## 📄 Licence

Ce projet est sous licence MIT — utilisable librement pour usage personnel et commercial.