"""
Script de fine-tuning YOLOv11 sur le dataset CADOT
"""
from ultralytics import YOLO
from pathlib import Path

# Configuration
DATA_YAML = Path("CADOT_Dataset/yolo/data.yaml")
MODEL_NAME = "yolo11l.pt"  # Modèle large YOLOv11. Alternatives: yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11x.pt
EPOCHS = 100
IMG_SIZE = 640
BATCH_SIZE = 16  # Ajustez selon votre GPU/RAM
PROJECT = "runs/detect"
NAME = "cadot_yolo11l"

def main():
    """Entraîne YOLOv11 sur le dataset CADOT"""
    
    # Vérifier que le fichier data.yaml existe
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Le fichier {DATA_YAML} n'existe pas !")
    
    print(f"📁 Dataset: {DATA_YAML}")
    print(f"🤖 Modèle: {MODEL_NAME}")
    print(f"🔄 Epochs: {EPOCHS}")
    print(f"📊 Image size: {IMG_SIZE}")
    print(f"📦 Batch size: {BATCH_SIZE}")
    print("\n" + "="*60)
    
    # Charger le modèle pré-entraîné
    model = YOLO(MODEL_NAME)
    
    # Entraîner le modèle
    results = model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        project=PROJECT,
        name=NAME,
        # Autres paramètres utiles :
        patience=50,           # Early stopping si pas d'amélioration après 50 epochs
        save=True,             # Sauvegarder les checkpoints
        device=0,              # GPU 0 (ou 'cpu' si pas de GPU)
        workers=8,             # Nombre de workers pour le chargement des données
        pretrained=True,       # Utiliser les poids pré-entraînés
        optimizer='auto',      # Optimiseur automatique
        verbose=True,          # Afficher les détails
        seed=42,               # Pour la reproductibilité
        deterministic=True,    # Mode déterministe
        single_cls=False,      # Multi-classes
        rect=False,            # Rectangulaire training
        cos_lr=False,          # Cosine learning rate scheduler
        close_mosaic=10,       # Désactiver mosaic les 10 derniers epochs
        resume=False,          # Reprendre l'entraînement (False = nouveau)
        amp=True,              # Automatic Mixed Precision
        fraction=1.0,          # Utiliser 100% du dataset
        profile=False,         # Profiler ONNX
        freeze=None,           # Couches à geler (None = aucune)
        # Augmentation de données
        lr0=0.01,              # Learning rate initial
        lrf=0.01,              # Learning rate final
        momentum=0.937,        # Momentum SGD
        weight_decay=0.0005,   # Weight decay
        warmup_epochs=3.0,     # Epochs de warmup
        warmup_momentum=0.8,   # Momentum de warmup
        warmup_bias_lr=0.1,    # LR de warmup pour les biais
        box=7.5,               # Poids de la loss box
        cls=0.5,               # Poids de la loss classification
        dfl=1.5,               # Poids de la loss DFL
        pose=12.0,             # Poids de la loss pose (si applicable)
        kobj=2.0,              # Poids de la loss keypoint obj (si applicable)
        label_smoothing=0.0,   # Label smoothing epsilon
        nbs=64,                # Nominal batch size
        hsv_h=0.015,           # Augmentation Hue
        hsv_s=0.7,             # Augmentation Saturation
        hsv_v=0.4,             # Augmentation Value
        degrees=0.0,           # Rotation (degrés)
        translate=0.1,         # Translation
        scale=0.5,             # Scale
        shear=0.0,             # Shear
        perspective=0.0,       # Perspective
        flipud=0.0,            # Flip vertical
        fliplr=0.5,            # Flip horizontal
        mosaic=1.0,            # Mosaic augmentation
        mixup=0.0,             # Mixup augmentation
        copy_paste=0.0,        # Copy paste augmentation
    )
    
    print("\n" + "="*60)
    print("✅ Entraînement terminé !")
    print(f"📊 Résultats sauvegardés dans: {PROJECT}/{NAME}")
    print(f"🎯 Meilleur modèle: {PROJECT}/{NAME}/weights/best.pt")
    
    # Valider le modèle
    print("\n" + "="*60)
    print("🔍 Validation du meilleur modèle...")
    metrics = model.val()
    
    print(f"\n📈 Métriques de validation:")
    print(f"   mAP50: {metrics.box.map50:.4f}")
    print(f"   mAP50-95: {metrics.box.map:.4f}")
    
    return results, metrics

if __name__ == "__main__":
    main()
