import cv2
import os
from pathlib import Path
from tqdm import tqdm

# --- CONFIGURATION ---
IMG_DIR = Path("CADOT_YOLO/train/images")
LBL_DIR = Path("CADOT_YOLO/train/labels")
OUTPUT_LORA_DIR = Path("lora_dataset/basketballfield") # Dossier de sortie

TARGET_CLASS_ID = 1  # 1 = Basketball field
MIN_SIZE = 20        # On ignore les objets trop petits/flous

#TARGET_CLASS_ID = 9  # 9 = Rond-point (Change ici pour Piscine ou autre)
#MIN_SIZE = 50        # On ignore les objets trop petits/flous

def crop_objects():
    if OUTPUT_LORA_DIR.exists():
        import shutil
        shutil.rmtree(OUTPUT_LORA_DIR)
    OUTPUT_LORA_DIR.mkdir(parents=True)

    print(f">>> Extraction des objets classe {TARGET_CLASS_ID} pour LoRA...")
    
    count = 0
    label_files = list(LBL_DIR.glob("*.txt"))

    for lbl_path in tqdm(label_files):
        # Lecture Label
        with open(lbl_path, 'r') as f:
            lines = f.readlines()
        
        # On cherche si l'objet cible est présent
        targets = []
        for line in lines:
            parts = line.split()
            cls_id = int(parts[0])
            if cls_id == TARGET_CLASS_ID:
                # x_c, y_c, w, h
                targets.append([float(x) for x in parts[1:]])
        
        if not targets: continue

        # Chargement Image
        img_path = IMG_DIR / (lbl_path.stem + ".jpg")
        if not img_path.exists(): continue
        
        img = cv2.imread(str(img_path))
        if img is None: continue
        h_img, w_img, _ = img.shape

        # Découpage (Crop)
        for i, bbox in enumerate(targets):
            xc, yc, w, h = bbox
            
            # Conversion YOLO -> Pixels
            x1 = int((xc - w / 2) * w_img)
            y1 = int((yc - h / 2) * h_img)
            x2 = int((xc + w / 2) * w_img)
            y2 = int((yc + h / 2) * h_img)
            
            # Sécurité bords
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_img, x2), min(h_img, y2)

            # Vérification taille
            if (x2 - x1) < MIN_SIZE or (y2 - y1) < MIN_SIZE:
                continue

            crop = img[y1:y2, x1:x2]
            
            # Sauvegarde
            # On ajoute un mot clé magique dans le nom pour aider l'entrainement plus tard
            save_name = f"basketballfield_{count}.jpg" 
            cv2.imwrite(str(OUTPUT_LORA_DIR / save_name), crop)
            count += 1

    print(f"\nTerminé ! {count} images extraites dans {OUTPUT_LORA_DIR}")

if __name__ == "__main__":
    crop_objects()