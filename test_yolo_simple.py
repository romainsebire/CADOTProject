#!/usr/bin/env python3
"""
Test simple YOLOv11n pour valider le fonctionnement MPS
"""
from ultralytics import YOLO
import os

# Optimisations MPS de base
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

print("🧪 Test YOLOv11n sur 10 époques pour validation MPS")
print("=" * 60)

# Charger le modèle nano (plus rapide pour test)
model = YOLO("yolo11n.pt")

print("🎯 Configuration test:")
print("├─ Modèle: Nano (2.6M paramètres)")
print("├─ Batch: 4 (minimal)")
print("├─ Workers: 0 (sécurisé)")
print("├─ Cache: False (économie RAM)")
print("└─ Époques: 10 (test rapide)")

# Test d'entraînement minimal
results = model.train(
    data="CADOT_Dataset/yolo/data.yaml",
    epochs=10,                              # TEST: seulement 10 époques
    imgsz=416,                              # RÉDUIT: 416 au lieu de 640
    batch=4,                                # MINIMAL: batch très petit
    name="test_yolo11n_mps",
    device="mps",
    workers=0,                              # Sécurisé
    cache=False,                            # Pas de cache
    amp=False,                              # Désactiver AMP
    verbose=True,                           # Verbeux
    patience=5,                             # Early stopping rapide
    save=True,
    # Désactiver augmentations lourdes
    copy_paste=0.0,
    mixup=0.0,
    mosaic=0.5,                             # Réduit
    cos_lr=False,                           # LR simple
)

print("\n" + "="*60)
print("✅ Test YOLOv11n terminé !")
print(f"📊 Résultats: {results}")

# Validation rapide
metrics = model.val()
print(f"\n📈 Métriques test:")
print(f"   mAP50: {metrics.box.map50:.4f}")
print(f"   mAP50-95: {metrics.box.map:.4f}")
print("\n🚀 Si ce test fonctionne, on peut passer à YOLOv11m !")