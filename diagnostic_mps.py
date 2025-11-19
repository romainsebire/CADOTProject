#!/usr/bin/env python3
"""
Diagnostic complet des problèmes MPS + YOLO
"""
import torch
from ultralytics import YOLO
import os
from pathlib import Path

print("🔍 DIAGNOSTIC COMPLET MPS + YOLO")
print("=" * 60)

# 1. Test MPS basique
print("\n1. 🧪 Test MPS basique:")
if torch.backends.mps.is_available():
    device = torch.device("mps")
    x = torch.rand(100, 100, device=device)
    y = x @ x.T
    print(f"   ✅ MPS fonctionne: tensor {x.device}")
    del x, y
else:
    print("   ❌ MPS non disponible")
    exit(1)

# 2. Test dataset YOLO
print("\n2. 📁 Test dataset:")
data_yaml = Path("CADOT_Dataset/yolo/data.yaml")
if data_yaml.exists():
    print(f"   ✅ data.yaml trouvé: {data_yaml}")
    
    # Lire le contenu
    import yaml
    with open(data_yaml) as f:
        data = yaml.safe_load(f)
    print(f"   📊 Classes: {data.get('nc', 'N/A')}")
    print(f"   📁 Train: {data.get('train', 'N/A')}")
    print(f"   📁 Val: {data.get('val', 'N/A')}")
else:
    print(f"   ❌ data.yaml non trouvé: {data_yaml}")
    exit(1)

# 3. Test modèle YOLO simple sans entraînement
print("\n3. 🤖 Test chargement modèle:")
try:
    model = YOLO("yolo11n.pt")
    print("   ✅ YOLOv11n chargé")
    
    # Test de prédiction simple
    print("   🎯 Test prédiction...")
    # Créer image test
    test_img = torch.rand(1, 3, 416, 416)
    # Pas de .to(device) pour éviter les problèmes
    print("   ✅ Test tensor créé")
    
except Exception as e:
    print(f"   ❌ Erreur modèle: {e}")

# 4. Test validation seule (sans entraînement)
print("\n4. 📊 Test validation (sans entraînement):")
try:
    model = YOLO("yolo11n.pt")
    print("   🔄 Validation en cours...")
    
    # Test validation avec CPU d'abord
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    
    # Validation avec paramètres sûrs
    metrics = model.val(
        data="CADOT_Dataset/yolo/data.yaml",
        device="cpu",  # CPU d'abord pour tester
        workers=0,
        batch=1,
        verbose=False
    )
    
    print(f"   ✅ Validation CPU réussie: mAP50 = {metrics.box.map50:.3f}")
    
    # Maintenant test avec MPS
    print("   🔄 Test validation MPS...")
    metrics_mps = model.val(
        data="CADOT_Dataset/yolo/data.yaml", 
        device="mps",
        workers=0,
        batch=1,
        verbose=False
    )
    
    print(f"   ✅ Validation MPS réussie: mAP50 = {metrics_mps.box.map50:.3f}")
    
except Exception as e:
    print(f"   ❌ Erreur validation: {e}")
    import traceback
    traceback.print_exc()

# 5. Test micro-entraînement (1 epoch, 1 batch)
print("\n5. 🧪 Test micro-entraînement:")
try:
    model = YOLO("yolo11n.pt")
    
    # Paramètres ultra-conservateurs
    result = model.train(
        data="CADOT_Dataset/yolo/data.yaml",
        epochs=1,                    # 1 seule epoch
        imgsz=320,                   # Très petit
        batch=1,                     # 1 seul batch
        device="mps",
        workers=0,
        cache=False,
        amp=False,
        verbose=False,               # Pas de spam
        patience=1,
        save=False,                  # Pas de sauvegarde
        plots=False,                 # Pas de plots
        name="micro_test",
        # Désactiver toutes les augmentations
        copy_paste=0.0,
        mixup=0.0,
        mosaic=0.0,
        degrees=0.0,
        translate=0.0,
        scale=0.0,
        fliplr=0.0,
        flipud=0.0,
    )
    
    print("   ✅ Micro-entraînement réussi!")
    
except Exception as e:
    print(f"   ❌ Erreur micro-entraînement: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("🏁 DIAGNOSTIC TERMINÉ")
print("Analysez les résultats ci-dessus pour identifier le problème.")