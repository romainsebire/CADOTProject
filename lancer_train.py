#from ultralytics import YOLO

# Chargement du modèle
#model = YOLO('yolov8m.pt') 

# Lancement de l'entrainement
#if __name__ == '__main__':
#    model.train(
#        data='cadot.yaml',
#        epochs=100,
#        imgsz=512,
#        batch=12,
#        device=0,
#        project='runs',
#        name='finetune_medium_v1',
#        patience=25,
#        verbose=True
#    )

from ultralytics import YOLO

# Chargement du modèle v11 Medium
model = YOLO('yolo11m.pt') 

if __name__ == '__main__':
    print(">>> Lancement du Benchmark YOLOv11 (Settings Concours)...")
    
    model.train(
        data='cadot.yaml',
        device=0,
        project='runs', 
        name='finetune_v11m_300e_albumentations',
        
        # Hyperparamètres du Concours
        epochs=300,          # Durée demandée
        batch=16,            # Taille demandée
        imgsz=512,           # Taille de tes images
        
        optimizer='SGD',     # Optimiseur imposé (Stochastic Gradient Descent)
        lr0=0.01,            # Learning Rate initial
        momentum=0.937,      # Momentum
        weight_decay=0.0005, # Pour éviter l'overfitting

        # Augmentations de données
        degrees=90,      # Rotation +/- 90 degrés
        flipud=0.5,      # Miroir vertical (50% de chance)
        scale=0.5,       # Zoom in/out (+/- 50%)
        mosaic=1.0,      # Mosaïque
        copy_paste=0.1,   # Copy-Paste (utile pour densifier les objets)
        
        # Paramètres de gestion
        patience=20,          # IMPORTANT : On désactive l'arrêt anticipé pour forcer les 300 époques
        verbose=True,        # Afficher les logs
        
        # Optionnel : Cosine LR scheduler est souvent utilisé par défaut avec ces réglages,
        # on le laisse actif (cos_lr=False par défaut, mais souvent True dans les papiers, 
        # tu peux laisser par défaut car le papier ne le précise pas explicitement).
    )