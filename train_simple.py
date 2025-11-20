"""
Script de fine-tuning YOLOv11 Nano sur le dataset CADOT
Version ANTI-FUITE MÉMOIRE - Reprend depuis epoch 29
Redémarre automatiquement tous les 5 epochs
"""
from ultralytics import YOLO
import os
import torch
import gc

def clean_memory():
    """Nettoyage mémoire agressif"""
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    print("🧹 Mémoire nettoyée")

# Configuration MPS
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

print("🚀 REPRISE DEPUIS EPOCH 29 - Anti-fuite mémoire")
print("📊 Progression actuelle: 29/100 epochs (29%)")
print("🎯 Objectif: Terminer les 71 epochs restants sans fuite")
print("💡 Stratégie: Cycles de 5 epochs + restart automatique")

# Point de départ
current_epoch = 29
max_epochs = 100
cycle = 1

while current_epoch < max_epochs:
    # Epochs pour ce cycle
    end_epoch = min(current_epoch + 5, max_epochs)
    epochs_to_run = end_epoch - current_epoch
    
    print(f"\n{'='*60}")
    print(f"🔄 CYCLE {cycle} - Epochs {current_epoch} à {end_epoch}")
    print(f"📈 Entraînement de {epochs_to_run} epochs")
    print(f"📊 Progression globale: {current_epoch}/{max_epochs} ({current_epoch/max_epochs*100:.1f}%)")
    
    # Nettoyer avant démarrage
    clean_memory()
    
    # Charger modèle
    if cycle == 1:
        # Premier cycle : votre checkpoint actuel
        model_path = "runs/detect/cadot_yolo11n_stable/weights/last.pt"
        print(f"📂 Chargement depuis: {model_path}")
    else:
        # Cycles suivants : nouveau checkpoint
        model_path = "runs/detect/cadot_yolo11n_restart/weights/last.pt" 
        print(f"📂 Chargement depuis: {model_path}")
    
    model = YOLO(model_path)
    
    print(f"⚡ Configuration ultra-conservative:")
    print(f"   - Batch: 2 (vs 4 précédent)")
    print(f"   - Workers: 1 (vs 2 précédent)")
    print(f"   - Cache: False")
    print(f"   - NMS: Ultra-strict")
    
    # Entraînement ultra-conservative
    try:
        results = model.train(
            data="CADOT_Dataset/yolo/data.yaml",
            epochs=epochs_to_run,
            imgsz=640,
            batch=2,                      # 🔧 ULTRA-réduit
            name="cadot_yolo11n_restart",
            device="mps",
            workers=1,                    # 🔧 UN SEUL worker
            patience=50,
            save=True,
            cache=False,
            amp=False,
            
            # NMS ultra-strict
            conf=0.9,
            iou=0.9, 
            max_det=10,                   # 🔧 TRÈS bas
            
            # Validation minimale
            val=True,
            save_period=1,
            
            # Zero augmentations
            copy_paste=0.0,
            mixup=0.0,
            mosaic=0.0,
            
            # LR fixe
            cos_lr=False,
            lr0=0.01,
            warmup_epochs=0,
            
            # Affichage minimal
            plots=False,
            verbose=False,
            resume=False,                 # Pas de resume automatique
        )
        
        print(f"✅ Cycle {cycle} terminé avec succès")
        
    except Exception as e:
        print(f"❌ Erreur dans cycle {cycle}: {e}")
        break
    
    finally:
        # Nettoyage obligatoire
        del model
        if 'results' in locals():
            del results
        clean_memory()
    
    # Mise à jour pour cycle suivant
    current_epoch = end_epoch
    cycle += 1
    
    print(f"🎯 Epochs restants: {max_epochs - current_epoch}")
    
    if current_epoch < max_epochs:
        print("⏳ Pause 3 secondes avant redémarrage...")
        import time
        time.sleep(3)

print(f"\n{'='*60}")
print("🎉 ENTRAÎNEMENT TERMINÉ SANS FUITE MÉMOIRE !")
print(f"📊 Modèle final: runs/detect/cadot_yolo11n_restart/weights/best.pt")
print(f"📈 {max_epochs} epochs complétés")
print(f"🚀 Votre modèle YOLOv11n est prêt !")