import cv2
import os
import numpy as np
from pathlib import Path
import random
from tqdm import tqdm

# --- CONFIGURATION ---
IMG_DIR = Path("CADOT_YOLO/train/images")
LBL_DIR = Path("CADOT_YOLO/train/labels")
OUTPUT_DIR = Path("1_inpainting_preparation")

# Parameters
CLASS_ID = 1      # Basketball ID
OBJ_W_PX = 100    # Desired width (in pixels, approx)
OBJ_H_PX = 70     # Desired height
COUNT = 15        # Number of images to prepare

def check_overlap(new_box, existing_boxes):
    """Checks if the new box overlaps with existing objects"""
    nx1, ny1, nx2, ny2 = new_box
    for bbox in existing_boxes:
        # bbox is in pixel format (x1, y1, x2, y2)
        ex1, ey1, ex2, ey2 = bbox
        
        # Intersection calculation
        ix1 = max(nx1, ex1)
        iy1 = max(ny1, ey1)
        ix2 = min(nx2, ex2)
        iy2 = min(ny2, ey2)
        
        if ix1 < ix2 and iy1 < iy2:
            return True # Overlap detected
    return False

def prepare_masks():
    if OUTPUT_DIR.exists():
        import shutil
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)
    (OUTPUT_DIR / "images").mkdir()
    (OUTPUT_DIR / "masks").mkdir()
    (OUTPUT_DIR / "coords").mkdir()

    print(f">>> Searching for empty spots for {COUNT} basketball courts...")
    
    img_files = list(IMG_DIR.glob("*.jpg"))
    random.shuffle(img_files) # Shuffle to vary backgrounds
    
    success_count = 0
    
    for img_path in tqdm(img_files):
        if success_count >= COUNT: break
        
        # 1. Read image and existing labels
        img = cv2.imread(str(img_path))
        if img is None: continue
        h_img, w_img, _ = img.shape
        
        lbl_path = LBL_DIR / (img_path.stem + ".txt")
        existing_boxes = []
        
        if lbl_path.exists():
            with open(lbl_path, 'r') as f:
                lines = f.readlines()
            for line in lines:
                parts = line.split()
                # YOLO -> Pixels conversion (x1, y1, x2, y2)
                xc, yc, w, h = map(float, parts[1:])
                x1 = int((xc - w/2) * w_img)
                y1 = int((yc - h/2) * h_img)
                x2 = int((xc + w/2) * w_img)
                y2 = int((yc + h / 2) * h_img)
                existing_boxes.append([x1, y1, x2, y2])

        # 2. Search for an empty spot (Random attempts)
        found_spot = False
        target_box = None
        
        for _ in range(50): # 50 attempts per image
            # 20px safety margin at edges
            x1 = random.randint(20, w_img - OBJ_W_PX - 20)
            y1 = random.randint(20, h_img - OBJ_H_PX - 20)
            x2 = x1 + OBJ_W_PX
            y2 = y1 + OBJ_H_PX
            
            new_box = [x1, y1, x2, y2]
            
            if not check_overlap(new_box, existing_boxes):
                found_spot = True
                target_box = new_box
                break
        
        if not found_spot: continue # Image too cluttered, skipping

        # 3. Create the mask (White on Black background)
        # This is required by tools like Stable Diffusion Inpainting
        mask = np.zeros((h_img, w_img), dtype=np.uint8)
        cv2.rectangle(mask, (target_box[0], target_box[1]), (target_box[2], target_box[3]), 255, -1)
        
        # 4. Save files
        basename = f"basket_gen_{success_count}_{img_path.stem}"
        
        # A. Original image (for upload)
        cv2.imwrite(str(OUTPUT_DIR / "images" / (basename + ".jpg")), img)
        
        # B. Mask (to tell AI where to draw)
        cv2.imwrite(str(OUTPUT_DIR / "masks" / (basename + ".png")), mask)
        
        # C. YOLO coordinates (for later)
        # Calculate normalized coordinates of the FUTURE object
        xc = (target_box[0] + OBJ_W_PX/2) / w_img
        yc = (target_box[1] + OBJ_H_PX/2) / h_img
        wn = OBJ_W_PX / w_img
        hn = OBJ_H_PX / h_img
        
        with open(OUTPUT_DIR / "coords" / (basename + ".txt"), 'w') as f:
            f.write(f"{CLASS_ID} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}\n")
            
        success_count += 1

    print(f"\n Ready! {success_count} generation kits in 1_inpainting_preparation/")
    print("Use the images in 'images/' and the masks in 'masks/' with your inpainting tool.")

if __name__ == "__main__":
    prepare_masks()