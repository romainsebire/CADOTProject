"""
Script de fine-tuning YOLOv11 Nano sur le dataset CADOT
Optimisé MPS pour Mac M2 16Go RAM - Version ULTRA-RAPIDE
"""
from ultralytics import YOLO
import os
import torch

# Optimisations MPS pour Mac M2
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

print("🍎 Optimisations MPS activées pour Mac M2")
print(f"🚀 MPS disponible: {torch.backends.mps.is_available()}")
print("🔥 Configuration ULTRA-RAPIDE: YOLOv11n + Batch 12")

# Charger le modèle pré-entraîné YOLOv11 Nano
model = YOLO("yolo11n.pt")

# Entraîner le modèle avec BATCH OPTIMISÉ
results = model.train(
    data="CADOT_Dataset/yolo/data.yaml",
    epochs=100,
    imgsz=640,
    batch=12,                              
    name="cadot_yolo11n_mps_ultra_fast",
    device="mps",
    workers=8,                             
    patience=25,
    save=True,
    cache=True,
    amp=False,
    close_mosaic=10,                      
    
    # ✅ PARAMÈTRES NMS OPTIMISÉS
    conf=0.6,
    iou=0.7,
    max_det=50,
    agnostic_nms=False,
    
    # ✅ OPTIMISATIONS ENTRAÎNEMENT
    val=True,
    save_period=20,
    copy_paste=0.0,
    mixup=0.0,
    mosaic=0.5,                            # 🚀 RÉDUIT de 0.7 à 0.5 (vitesse)
    cos_lr=True,
    warmup_epochs=2,                       # 🚀 RÉDUIT de 3 à 2 (vitesse)
    warmup_momentum=0.5,
    
    # ✅ AFFICHAGE MINIMAL
    plots=False,
    verbose=False,
)

print("\n" + "="*60)
print("🔥 Entraînement YOLOv11n ULTRA-RAPIDE terminé !")
print(f"📊 Modèle final: runs/detect/cadot_yolo11n_mps_ultra_fast/weights/best.pt")

# Validation finale optimisée
metrics = model.val(
    conf=0.6,
    iou=0.7,
    max_det=50,
    plots=False,
    verbose=True
)