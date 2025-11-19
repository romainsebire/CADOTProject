# CADOT Project - Détection d'objets sur images aériennes de Paris

## 📋 Description du projet

Ce projet vise à améliorer la détection d'objets sur des images aériennes de Paris issues du défi [cadot.onrender.com](https://cadot.onrender.com). 

**Workflow du projet :**

1. **Fine-tuning YOLOv8** sur l'ensemble des images annotées pour obtenir une détection robuste des catégories d'objets présentes en milieu urbain dense
2. **Analyse des performances** pour identifier les classes les moins bien détectées (faible précision, rappel ou mAP)
3. **Génération de données synthétiques** via modèles de diffusion (Stable Diffusion + LoRA) pour les objets rares ou difficiles
4. **Second fine-tuning** avec les données augmentées pour améliorer la performance globale

## 📊 Dataset

**Statistiques du dataset CADOT :**
- **Total images** : 4,163 (3,234 train / 929 validation)
- **Total annotations** : 96,640 boîtes englobantes
- **Classes** : 15 catégories d'objets urbains
- **Moyenne** : 23.21 objets par image

**Distribution des classes (top 5) :**
- Small vehicle : 46.7%
- Building : 40.9%
- Medium vehicle : 5.1%
- Crosswalk : 5.0%
- Large vehicle : 1.1%

## 🚀 Installation

```bash
# Cloner le repo
git clone https://github.com/romainsebire/CADOTProject.git
cd CADOTProject

# Installer les dépendances
pip install -r requirements.txt
```

## 🎯 Quick Start

### 1. Vérifier le dataset

```bash
python check_dataset.py
```

### 2. Entraîner YOLOv8 (version simple)

```bash
python train_simple.py
```

### 3. Entraîner YOLOv8 (version complète avec tous les paramètres)

```bash
python train_yolo.py
```

Voir le [Guide d'entraînement détaillé](TRAINING_GUIDE.md) pour plus d'informations.

## 📂 Structure du projet

```
CADOTProject/
├── CADOT_Dataset/
│   ├── train/                    # Images et annotations COCO originales
│   ├── valid/                    # Images et annotations COCO de validation
│   └── yolo/
│       ├── data.yaml             # Configuration dataset YOLO
│       ├── images/
│       │   ├── train/            # Images train
│       │   └── valid/            # Images valid
│       └── labels/
│           ├── train/            # Annotations YOLO train
│           └── valid/            # Annotations YOLO valid
├── convert_coco_to_yolo.py       # Script de conversion COCO → YOLO
├── check_dataset.py              # Vérification du dataset
├── train_simple.py               # Script d'entraînement simple
├── train_yolo.py                 # Script d'entraînement complet
├── requirements.txt              # Dépendances Python
├── TRAINING_GUIDE.md             # Guide détaillé d'entraînement
└── README.md                     # Ce fichier
```

## 🔧 Configuration

Le fichier `CADOT_Dataset/yolo/data.yaml` configure tout le dataset :

```yaml
path: /chemin/vers/CADOT_Dataset/yolo  # Chemin absolu
train: images/train                     # Chemin relatif
val: images/valid                       # Chemin relatif
nc: 15                                  # Nombre de classes
names:                                  # Liste des classes
  - small-object
  - basketball field
  - building
  # ... etc
```

## 📈 Résultats

Les résultats d'entraînement sont sauvegardés dans `runs/detect/` :
- Métriques de performance (mAP, precision, recall)
- Graphiques d'entraînement
- Matrice de confusion
- Modèles entraînés (`best.pt` et `last.pt`)

## 🤝 Contribution

Ce projet est développé dans le cadre du défi CADOT pour l'amélioration de la détection d'objets sur images aériennes.

## 📚 Ressources

- [Documentation Ultralytics YOLOv8](https://docs.ultralytics.com/)
- [Défi CADOT](https://cadot.onrender.com)
- [Format de données YOLO](https://docs.ultralytics.com/datasets/detect/)