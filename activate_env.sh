#!/bin/bash
# Script d'activation de l'environnement virtuel CADOT Project

echo "🚀 Activation de l'environnement virtuel CADOT Project..."

# Ajouter uv au PATH
export PATH="$HOME/.local/bin:$PATH"

# Activer l'environnement virtuel
source .venv/bin/activate

echo "✅ Environnement virtuel activé !"
echo "📦 Dépendances installées avec uv"
echo ""
echo "Commandes disponibles :"
echo "  python train_simple.py    # Entraînement optimisé M2"
echo "  python train_yolo.py      # Entraînement avec plus d'options"
echo "  python convert_coco_to_yolo.py  # Conversion dataset"
echo ""
echo "Pour désactiver : deactivate"