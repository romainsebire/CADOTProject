from ultralytics import YOLO

# Load v11 Medium model
model = YOLO('yolo11m.pt') 

if __name__ == '__main__':
    
    model.train(
        data='cadot.yaml',
        device=0,
        project='runs', 
        name='finetune_v11m', # Output folder name in /runs/..
        
        # Competition Hyperparameters
        epochs=300,          # Required epochs
        batch=16,            # Required batch size
        imgsz=512,           # Image size (500 x 500)
        
        optimizer='SGD',     # Mandated optimizer (Stochastic Gradient Descent)
        lr0=0.01,            # Initial Learning Rate
        momentum=0.937,      # Momentum
        weight_decay=0.0005, # To prevent overfitting

        # Data Augmentations
        degrees=90,      # Rotation +/- 90 degrees
        flipud=0.5,      # Vertical flip (50% probability)
        scale=0.5,       # Zoom in/out (+/- 50%)
        mosaic=1.0,      # Mosaic
        copy_paste=0.1,  # Copy-Paste (useful to densify objects)
        
        # Management settings
        patience=20,         # Early stopping if no progress
        verbose=True,        # Display logs
    )