import shutil
import os
from pathlib import Path

# --- CONFIGURATION ---
GENERATED_DIR = Path("GENERATED_RESULTS") # Where you put your finished images
COORD_DIR = Path("1_inpainting_preparation/coords") # Where script prepare_inpainting put the info

FINAL_IMG_DIR = Path("CADOT_YOLO/data_augmentation/images")
FINAL_LBL_DIR = Path("CADOT_YOLO/data_augmentation/labels")
ORIGINAL_LBL_DIR = Path("CADOT_YOLO/train/labels") # To retrieve background labels

def merge_data():
    print(">>> Merging generated images into the dataset...")
    
    gen_files = list(GENERATED_DIR.glob("*.jpg"))
    
    count = 0
    for gen_img_path in gen_files:
        # The filename must match the one generated in step 1
        # Be careful if your tool renames files, rename them first!
        stem = gen_img_path.stem
        
        # 1. Retrieve saved coordinates
        coord_path = COORD_DIR / (stem + ".txt")
        if not coord_path.exists():
            print(f"⚠️ No coordinates found for {stem}, skipped.")
            continue
            
        # 2. Retrieve original labels (other objects in the background image)
        # Extract original image name from compound name
        original_stem = stem.split('_', 3)[3] 
        original_lbl_path = ORIGINAL_LBL_DIR / (original_stem + ".txt")
        
        existing_labels = []
        if original_lbl_path.exists():
            with open(original_lbl_path, 'r') as f:
                existing_labels = f.readlines()
        
        # 3. Create the new complete label file
        final_lbl_path = FINAL_LBL_DIR / (stem + ".txt")
        
        with open(final_lbl_path, 'w') as f_out:
            # A. Old objects
            for line in existing_labels:
                f_out.write(line)
            
            # B. The new basketball court
            with open(coord_path, 'r') as f_coord:
                new_obj_line = f_coord.read()
                f_out.write(new_obj_line)
        
        # 4. Copy the generated image to the Train folder
        shutil.copy(gen_img_path, FINAL_IMG_DIR / (stem + ".jpg"))
        
        count += 1

    print(f"\n Finished! {count} new synthetic images added to the dataset.")

if __name__ == "__main__":
    merge_data()