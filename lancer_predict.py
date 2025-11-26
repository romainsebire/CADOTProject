from ultralytics import YOLO
import os

# 1. Chemin vers ton meilleur modèle (le Medium v11 entraîné)
# Vérifie bien que ce chemin correspond à ce que tu as dans tes logs précédents
model_path = 'runs/finetune_v11m_300e/weights/best.pt'

# 2. Chemin vers les images de TEST (celles sans labels)
source_images = 'CADOT_YOLO/test/images'

# Chargement
if not os.path.exists(model_path):
    print(f"ERREUR : Modèle introuvable ici : {model_path}")
    print("Vérifie dans ton dossier 'runs' où est rangé 'best.pt'")
    exit()

model = YOLO(model_path)

print(f">>> Lancement des prédictions sur {source_images}...")

# 3. Prédiction
model.predict(
    source=source_images,
    save=True,           # Sauvegarder les images avec les boîtes
    conf=0.25,           # Seuil de confiance (25%)
    iou=0.45,            # Seuil pour éviter les boîtes en double
    project='runs',      # Dossier racine
    name='predict_test', # Nom du dossier de sortie
    exist_ok=True        # Écrase si le dossier existe déjà
)

print("\nPrédictions terminées !")
print("Les images sont dans : runs/predict_test")