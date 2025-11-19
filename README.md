# CADOT Project - Détection d'objets sur images aériennes de Paris

## 📋 Description du projet

Ce projet vise à améliorer la détection d'objets sur des images aériennes de Paris issues du défi [cadot.onrender.com](https://cadot.onrender.com). 

**Workflow du projet :**

1. **Fine-tuning YOLOv11** sur l'ensemble des images annotées pour obtenir une détection robuste des catégories d'objets présentes en milieu urbain dense
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
pip3 install -r requirements.txt
```

## 🎯 Quick Start

### Entraînement YOLOv11l sur Mac M2 16Go

```bash
# Lancer l'entraînement (recommandé pour M2 16Go)
python3 train_simple.py
```

**Configuration optimisée :**
- Modèle : YOLOv11l (Large - 43.7M paramètres)
- Device : MPS (Metal Performance Shaders)
- Batch size : 8
- Epochs : 100
- Image size : 640
- Cache : Activé (utilise la RAM pour accélérer)
- Workers : 8
- Durée estimée : ~30 heures

**Script alternatif avec tous les paramètres configurables :**
```bash
python3 train_yolo.py
```

## 📂 Structure du projet

```
CADOTProject/
├── CADOT_Dataset/
│   ├── train/                    # Images et annotations COCO originales
│   ├── valid/                    # Images et annotations COCO de validation
│   └── yolo/
│       ├── data.yaml             # Configuration dataset YOLO
│       ├── images/
│       │   ├── train/            # Images train (3,234 images)
│       │   └── valid/            # Images valid (929 images)
│       └── labels/
│           ├── train/            # Annotations YOLO train (format txt)
│           └── valid/            # Annotations YOLO valid (format txt)
├── convert_coco_to_yolo.py       # Script de conversion COCO → YOLO
├── train_simple.py               # Script d'entraînement YOLOv11l (optimisé M2)
├── train_yolo.py                 # Script d'entraînement avec paramètres avancés
├── requirements.txt              # Dépendances Python
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

Les résultats d'entraînement sont sauvegardés dans `runs/detect/cadot_yolo11l/` :
- **Métriques** : mAP50, mAP50-95, precision, recall par classe
- **Graphiques** : courbes de loss, métriques d'entraînement
- **Matrice de confusion** : analyse des erreurs de classification
- **Modèles** : `best.pt` (meilleur mAP) et `last.pt` (dernière époque)
- **Prédictions** : visualisations des détections sur validation set

## 🤝 Contribution

Ce projet est développé dans le cadre du défi CADOT pour l'amélioration de la détection d'objets sur images aériennes.

## 🖥️ Matériel recommandé

**Configuration utilisée :**
- **CPU** : Apple M2
- **RAM** : 16 Go
- **GPU** : Metal Performance Shaders (MPS)
- **Stockage** : ~5 Go pour dataset + modèles

**Optimisations Apple Silicon :**
- Accélération GPU via MPS (PyTorch 2.0+)
- Cache dataset en RAM pour I/O rapide
- AMP désactivé (incompatibilité MPS)

## 📚 Ressources

- [Documentation Ultralytics YOLO](https://docs.ultralytics.com/)
- [YOLOv11 Model Hub](https://docs.ultralytics.com/models/yolo11/)
- [Défi CADOT](https://cadot.onrender.com)
- [Format de données YOLO](https://docs.ultralytics.com/datasets/detect/)
- [PyTorch MPS Backend](https://pytorch.org/docs/stable/notes/mps.html)


Commandes Mac M2 : 
cd /Users/remyplastre/Documents/GitHub/CADOTProject
git pull
pip3 install -r requirements.txt
python3 train_simple.py