import cv2
import os
from pathlib import Path
from tqdm import tqdm
import random

# --- CONFIGURATION ---
IMG_DIR = Path("CADOT_YOLO/data_augmentation/images")
LBL_DIR = Path("CADOT_YOLO/data_augmentation/labels")
OUTPUT_DIR = Path("CADOT_YOLO/data_augmentation/visual_debug")

# Class Names
CLASS_NAMES = {
    0: "Small Object", 1: "Basketball", 2: "Building", 3: "Crosswalk",
    4: "Football", 5: "Graveyard", 6: "Large Veh", 7: "Medium Veh",
    8: "Playground", 9: "Roundabout", 10: "Ship", 11: "Small Veh",
    12: "Pool", 13: "Tennis", 14: "Train"
}

# Fixed colors for each class (B, G, R) for readability
COLORS = {
    1: (0, 255, 0),   # Basketball = Neon Green
    9: (0, 255, 255), # Roundabout = Yellow
    12: (255, 0, 255),# Pool = Magenta
}

def visualize_all():
    # 1. Create output directory
    if OUTPUT_DIR.exists():
        import shutil
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 2. List images
    print(f">>> Scanning images in {IMG_DIR}...")
    if not IMG_DIR.exists():
        print("Image directory not found!")
        return

    img_files = list(IMG_DIR.glob("*.jpg")) + list(IMG_DIR.glob("*.jpeg"))
    print(f"   {len(img_files)} images found.")

    count_processed = 0

    for img_path in tqdm(img_files):
        # Find associated label
        lbl_path = LBL_DIR / (img_path.stem + ".txt")
        
        if not lbl_path.exists():
            continue # Skip images without labels

        # Load image
        img = cv2.imread(str(img_path))
        if img is None: continue
        h_img, w_img, _ = img.shape

        # Read annotations
        with open(lbl_path, 'r') as f:
            lines = f.readlines()

        has_objects = False
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5: continue
            
            try:
                # Robust reading (handles "4" and "4.0")
                cls_id = int(float(parts[0]))
                xc, yc, w, h = map(float, parts[1:])

                # YOLO -> Pixels Conversion
                x1 = int((xc - w / 2) * w_img)
                y1 = int((yc - h / 2) * h_img)
                x2 = int((xc + w / 2) * w_img)
                y2 = int((yc + h / 2) * h_img)

                # Color (Specific or Red by default)
                color = COLORS.get(cls_id, (0, 0, 255)) 
                
                # Draw rectangle
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                
                # Add text (Black background for readability)
                label = f"{cls_id}: {CLASS_NAMES.get(cls_id, '')}"
                (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(img, (x1, y1 - text_h - 5), (x1 + text_w, y1), color, -1) # Colored background
                cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
                
                has_objects = True
            except ValueError:
                pass

        # Save debug image
        if has_objects:
            cv2.imwrite(str(OUTPUT_DIR / img_path.name), img)
            count_processed += 1

    print(f"\nFinished! {count_processed} verification images generated.")
    print(f"Directory: {OUTPUT_DIR}")

if __name__ == "__main__":
    visualize_all()