import albumentations as A
import cv2
import os
from pathlib import Path
from tqdm import tqdm

# --- CONFIGURATION ---
IMG_DIR = Path("CADOT_YOLO/train/images")
LBL_DIR = Path("CADOT_YOLO/train/labels")

# --- 1. QUANTITY FACTORS (How many images do we want?) ---
augment_counts = {
    1: 35,   # Basketball
    9: 15,   # Roundabout
    8: 12,   # Playground
    12: 8,   # Swimming Pool
    14: 8,   # Train
    4:  7,   # Football (Soccer)
    13: 5,   # Tennis
    5:  4,   # Graveyard
    10: 1,   # Ship
}

# --- 2. SPECIFIC PIPELINES DEFINITION ---

# A. SPORT Pipeline (Sacred Color and Lines)
# Target: Basket(1), Foot(4), Tennis(13), Crosswalk(3)
pipe_sport = A.Compose([
    A.RandomRotate90(p=1.0),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    # We play with light (sun/shadow) but NOT color (Hue)
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.8),
    # No Blur here!
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# B. TEXTURE Pipeline (Sacred Details)
# Target: Swimming Pool(12), Graveyard(5)
pipe_texture = A.Compose([
    A.RandomRotate90(p=1.0),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    # Enhances details (waves, stones)
    A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.5),
    A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5), # Strong local contrast
    # BLUR FORBIDDEN
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# C. SHAPE Pipeline (Flexible Geometry)
# Target: Roundabout(9), Playground(8), Train(14), Ship(10)
pipe_shape = A.Compose([
    A.RandomRotate90(p=1.0),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    # Slight distortion to change curvature or viewing angle
    A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
    A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2),
    # Here blur is accepted (speed or atmosphere)
    A.GaussianBlur(blur_limit=(3, 5), p=0.2),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))


# --- MAPPING: Which class uses which pipeline? ---
CLASS_PIPELINES = {
    1: pipe_sport,    # Basket
    4: pipe_sport,    # Football
    13: pipe_sport,   # Tennis
    
    5: pipe_texture,  # Graveyard
    12: pipe_texture, # Swimming Pool
    
    9: pipe_shape,    # Roundabout
    8: pipe_shape,    # Playground
    14: pipe_shape,   # Train
    10: pipe_shape,   # Ship
    6: pipe_shape     # Large Vehicle (if augmenting)
}

def get_best_pipeline_and_factor(present_ids):
    """
    Determines the pipeline and augmentation factor based on 
    the most critical object present in the image.
    """
    target_factor = 0
    selected_pipeline = None
    
    # We look for the object with the highest factor (the rarest)
    for cls in present_ids:
        if cls in augment_counts:
            if augment_counts[cls] > target_factor:
                target_factor = augment_counts[cls]
                # We select the pipeline associated with this critical object
                selected_pipeline = CLASS_PIPELINES.get(cls, pipe_shape) # pipe_shape by default
    
    return target_factor, selected_pipeline

def augment_expert_data():

    label_files = list(LBL_DIR.glob("*.txt"))
    count_new = 0
    
    for lbl_path in tqdm(label_files):
        # 1. Reading
        with open(lbl_path, 'r') as f:
            lines = f.readlines()
        
        present_classes = []
        bboxes = []
        class_labels = []
        
        for line in lines:
            parts = line.strip().split()
            cls_id = int(parts[0])
            coords = [float(x) for x in parts[1:]]
            
            present_classes.append(cls_id)
            bboxes.append(coords)
            class_labels.append(cls_id)
        
        # 2. Strategy selection
        factor, pipeline = get_best_pipeline_and_factor(present_classes)
        
        if factor == 0 or pipeline is None:
            continue
            
        # 3. Image Loading
        img_name = lbl_path.stem + ".jpg"
        img_path = IMG_DIR / img_name
        if not img_path.exists(): continue
            
        image = cv2.imread(str(img_path))
        if image is None: continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 4. Generation
        for i in range(factor):
            try:
                augmented = pipeline(image=image, bboxes=bboxes, class_labels=class_labels)
                
                # Save
                new_filename = f"{lbl_path.stem}_expert_{i}"
                save_img_path = IMG_DIR / (new_filename + ".jpg")
                save_lbl_path = LBL_DIR / (new_filename + ".txt")
                
                img_to_save = cv2.cvtColor(augmented['image'], cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(save_img_path), img_to_save)
                
                with open(save_lbl_path, 'w') as f_out:
                    for cls, bbox in zip(augmented['class_labels'], augmented['bboxes']):
                        xc, yc, w, h = bbox
                        xc, yc = min(max(xc, 0), 1), min(max(yc, 0), 1)
                        w, h = min(max(w, 0), 1), min(max(h, 0), 1)
                        f_out.write(f"{int(cls)} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
                
                count_new += 1
            except Exception:
                pass

    print(f"\nFinished! {count_new} images generated.")

if __name__ == "__main__":
    augment_expert_data()