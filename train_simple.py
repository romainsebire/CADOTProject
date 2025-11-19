"""
Script de fine-tuning YOLOv11 Large sur le dataset CADOT
Optimisé pour Mac M2 16Go RAM - Entraînement sur 48h
"""
from ultralytics import YOLO

# Charger le modèle pré-entraîné YOLOv11 Large
model = YOLO("yolo11m.pt")

# Entraîner le modèle
results = model.train(
    data="CADOT_Dataset/yolo/data.yaml",  # Chemin vers votre fichier de config
    epochs=100,                            # Nombre d'epochs
    imgsz=640,                             # Taille des images
    batch=8,                               # Optimal pour M2 16Go avec Large
    name="cadot_yolo11m",                  # Nom de l'expérience
    device="mps",                          # GPU M2 via Metal Performance Shaders
    workers=8,                             # Plus de workers pour M2
    patience=25,                           # Patience pour early stopping
    save=True,                             # Sauvegarder les checkpoints
    cache=True,                            # Mettre les images en cache (16Go RAM!)
    amp=False,                             # Désactiver AMP pour MPS
    close_mosaic=10,                       # Améliore la précision finale
)

# Validation finale
print("\n" + "="*60)
print("✅ Entraînement terminé !")
print(f"📊 Meilleur modèle: runs/detect/cadot_yolo11m/weights/best.pt")
metrics = model.val()
print(f"📈 mAP50: {metrics.box.map50:.4f}")
print(f"📈 mAP50-95: {metrics.box.map:.4f}")

# Valider le modèle
metrics = model.val()

print(f"mAP50: {metrics.box.map50}")
print(f"mAP50-95: {metrics.box.map}")
