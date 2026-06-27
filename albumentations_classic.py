import albumentations as A
import cv2
import os
from pathlib import Path
from tqdm import tqdm

# --- CONFIGURATION ---
IMG_DIR = Path("CADOT_YOLO/train/images")
LBL_DIR = Path("CADOT_YOLO/train/labels")

# IDs of rare classes
# 1:Basket, 4:Foot, 5:Graveyard, 9:Roundabout, 13:Tennis, 14:Train
RARE_CLASSES = [1, 4, 5, 8, 9, 13, 14] 

# Augmentation factor
AUGMENT_FACTOR = 5 

# Albumentations Pipeline (Aerial specific)
transform = A.Compose([
    A.RandomRotate90(p=1.0),            # Rotation 90/180/270
    A.HorizontalFlip(p=0.5),            # Horizontal Mirror
    A.VerticalFlip(p=0.5),              # Vertical Mirror
    A.RandomBrightnessContrast(p=0.5),  # Lighting changes
    A.GaussianBlur(p=0.3),              # Slight blur
    A.CLAHE(p=0.3),                     # Adaptive contrast
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

def augment_data():
    label_files = list(LBL_DIR.glob("*.txt"))
    count_new = 0
    
    for lbl_path in tqdm(label_files):
        with open(lbl_path, 'r') as f:
            lines = f.readlines()
        
        has_rare = False
        bboxes = []
        class_labels = []
        
        for line in lines:
            parts = line.strip().split()
            cls_id = int(parts[0])
            coords = [float(x) for x in parts[1:]] # x_c, y_c, w, h
            
            bboxes.append(coords)
            class_labels.append(cls_id)
            
            if cls_id in RARE_CLASSES:
                has_rare = True
        
        if not has_rare:
            continue
        
        # Load Image
        img_name = lbl_path.stem + ".jpg"
        img_path = IMG_DIR / img_name
        
        if not img_path.exists():
            continue
            
        image = cv2.imread(str(img_path))
        if image is None: continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Generate Variations
        for i in range(AUGMENT_FACTOR):
            try:
                augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
                
                # Save Image
                new_filename = f"{lbl_path.stem}_aug_{i}"
                save_img_path = IMG_DIR / (new_filename + ".jpg")
                
                img_to_save = cv2.cvtColor(augmented['image'], cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(save_img_path), img_to_save)
                
                # Save Label
                save_lbl_path = LBL_DIR / (new_filename + ".txt")
                with open(save_lbl_path, 'w') as f_out:
                    for cls, bbox in zip(augmented['class_labels'], augmented['bboxes']):
                        xc, yc, w, h = bbox
                        xc = min(max(xc, 0), 1)
                        yc = min(max(yc, 0), 1)
                        w = min(max(w, 0), 1)
                        h = min(max(h, 0), 1)
                        f_out.write(f"{int(cls)} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
                
                count_new += 1
            except Exception as e:
                pass

    print(f"\n Finished! {count_new} new images generated for rare classes.")

if __name__ == "__main__":
    augment_data()