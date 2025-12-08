import cv2
import os
import numpy as np
from pathlib import Path
import random
from tqdm import tqdm

# --- CONFIGURATION ---
IMG_DIR = Path("CADOT_YOLO/train_original/images")
LBL_DIR = Path("CADOT_YOLO/train_original/labels")
OUTPUT_DIR = Path("INPAINTING_STAGING") # Dossier de travail

# Paramètres pour le Terrain de Basket
CLASS_ID = 1      # ID du Basket
OBJ_W_PX = 100    # Largeur voulue (en pixels, approx)
OBJ_H_PX = 70     # Hauteur voulue
COUNT = 15        # Nombre d'images à préparer

def check_overlap(new_box, existing_boxes):
    """Vérifie si la nouvelle boîte chevauche des objets existants"""
    nx1, ny1, nx2, ny2 = new_box
    for bbox in existing_boxes:
        # bbox est en format pixel (x1, y1, x2, y2)
        ex1, ey1, ex2, ey2 = bbox
        
        # Calcul de l'intersection
        ix1 = max(nx1, ex1)
        iy1 = max(ny1, ey1)
        ix2 = min(nx2, ex2)
        iy2 = min(ny2, ey2)
        
        if ix1 < ix2 and iy1 < iy2:
            return True # Il y a chevauchement
    return False

def prepare_masks():
    if OUTPUT_DIR.exists():
        import shutil
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)
    (OUTPUT_DIR / "images").mkdir()
    (OUTPUT_DIR / "masks").mkdir()
    (OUTPUT_DIR / "coords").mkdir()

    print(f">>> Recherche de zones vides pour {COUNT} terrains de basket...")
    
    img_files = list(IMG_DIR.glob("*.jpg"))
    random.shuffle(img_files) # On mélange pour varier les fonds
    
    success_count = 0
    
    for img_path in tqdm(img_files):
        if success_count >= COUNT: break
        
        # 1. Lire image et labels existants
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
                # Conversion YOLO -> Pixels (x1, y1, x2, y2)
                xc, yc, w, h = map(float, parts[1:])
                x1 = int((xc - w/2) * w_img)
                y1 = int((yc - h/2) * h_img)
                x2 = int((xc + w/2) * w_img)
                y2 = int((yc + h/2) * h_img)
                existing_boxes.append([x1, y1, x2, y2])

        # 2. Chercher un emplacement vide (Tentatives aléatoires)
        found_spot = False
        target_box = None
        
        for _ in range(50): # 50 essais par image
            # Marge de sécurité de 20px aux bords
            x1 = random.randint(20, w_img - OBJ_W_PX - 20)
            y1 = random.randint(20, h_img - OBJ_H_PX - 20)
            x2 = x1 + OBJ_W_PX
            y2 = y1 + OBJ_H_PX
            
            new_box = [x1, y1, x2, y2]
            
            if not check_overlap(new_box, existing_boxes):
                found_spot = True
                target_box = new_box
                break
        
        if not found_spot: continue # Image trop encombrée, on passe

        # 3. Créer le masque (Blanc sur fond Noir)
        # C'est ce que demandent les outils comme Stable Diffusion Inpainting
        mask = np.zeros((h_img, w_img), dtype=np.uint8)
        cv2.rectangle(mask, (target_box[0], target_box[1]), (target_box[2], target_box[3]), 255, -1)
        
        # 4. Sauvegarder les fichiers
        basename = f"basket_gen_{success_count}_{img_path.stem}"
        
        # A. L'image originale (pour l'upload)
        cv2.imwrite(str(OUTPUT_DIR / "images" / (basename + ".jpg")), img)
        
        # B. Le masque (pour dire à l'IA où dessiner)
        cv2.imwrite(str(OUTPUT_DIR / "masks" / (basename + ".png")), mask)
        
        # C. Les coordonnées YOLO (pour plus tard)
        # On calcule les coordonnées normalisées du FUTUR objet
        xc = (target_box[0] + OBJ_W_PX/2) / w_img
        yc = (target_box[1] + OBJ_H_PX/2) / h_img
        wn = OBJ_W_PX / w_img
        hn = OBJ_H_PX / h_img
        
        with open(OUTPUT_DIR / "coords" / (basename + ".txt"), 'w') as f:
            f.write(f"{CLASS_ID} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}\n")
            
        success_count += 1

    print(f"\n Prêt ! {success_count} kits de génération dans INPAINTING_STAGING/")
    print("Utilise les images dans 'images/' et les masques dans 'masks/' sur Nano Banana.")

if __name__ == "__main__":
    prepare_masks()