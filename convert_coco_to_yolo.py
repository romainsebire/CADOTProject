import json
import shutil
from pathlib import Path
from tqdm import tqdm

# --- CONFIGURATION ---
SOURCE_DIR = Path("Dataset_COCO")
OUTPUT_DIR = Path("Dataset_YOLO")

FOLDERS_MAP = {
    "train": "train",
    "valid": "val"
}

def convert_dataset():
    # 1. Cleanup and Creation of the output directory
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    
    OUTPUT_DIR.mkdir(parents=True)

    # 2. Loop over TRAIN and VAL (Those with labels)
    for src_name, dest_name in FOLDERS_MAP.items():
        print(f"\nProcessing '{src_name}' to '{dest_name}'...")
        
        src_path = SOURCE_DIR / src_name
        dest_img_dir = OUTPUT_DIR / dest_name / "images"
        dest_lbl_dir = OUTPUT_DIR / dest_name / "labels"
        dest_img_dir.mkdir(parents=True, exist_ok=True)
        dest_lbl_dir.mkdir(parents=True, exist_ok=True)


        json_file = list(src_path.glob("*json"))
        if not json_file:
            print(f"No JSON found in {src_name}! Skipping.")
            continue
        
        json_path = json_file[0] # Take the first one found

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        images_dict = {item['id']: item for item in data['images']}
        img_to_anns = {item['id']: [] for item in data.get('annotations', [])}
        for ann in data.get('annotations', []):
            img_to_anns[ann['image_id']].append(ann)

        for img_id, img_info in tqdm(images_dict.items()):
            filename = Path(img_info['file_name']).name
            src_img = src_path / filename

            if not src_img.exists():
                continue

            # Copy Image
            shutil.copy2(src_img, dest_img_dir / filename)

            # Create Label .txt
            txt_name = Path(filename).stem + ".txt"
            img_w, img_h = img_info['width'], img_info['height']
            
            with open(dest_lbl_dir / txt_name, 'w') as f_out:
                for ann in img_to_anns.get(img_id, []):
                    cls_id = ann['category_id']
                    x, y, w, h = ann['bbox']
                    
                    # Normalization
                    x_c, y_c = (x + w/2)/img_w, (y + h/2)/img_h
                    w_n, h_n = w/img_w, h/img_h
                    
                    # Clip (safety)
                    x_c, y_c = max(0, min(1, x_c)), max(0, min(1, y_c))
                    w_n, h_n = max(0, min(1, w_n)), max(0, min(1, h_n))
                    
                    f_out.write(f"{cls_id} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}\n")

    src_test = SOURCE_DIR / "test"
    dest_test = OUTPUT_DIR / "test" / "images"
    dest_test.mkdir(parents=True, exist_ok=True)
    
    if src_test.exists():
        for img in tqdm(list(src_test.glob("*.jpg")) + list(src_test.glob("*.jpeg"))):
            shutil.copy2(img, dest_test / img.name)
    
    print("\nFinished! 'Dataset_YOLO' done.")

if __name__ == "__main__":
    convert_dataset()