import cv2
import os
from pathlib import Path
from tqdm import tqdm
import random

# --- CONFIGURATION ---
IMG_DIR = Path("CADOT_YOLO/data_augmentation/images")
LBL_DIR = Path("CADOT_YOLO/data_augmentation/labels")
OUTPUT_DIR = Path("CADOT_YOLO/data_augmentation/visual_debug")

# Noms des classes (pour affichage)
CLASS_NAMES = {
    0: "Small Object", 1: "Basketball", 2: "Building", 3: "Crosswalk",
    4: "Football", 5: "Graveyard", 6: "Large Veh", 7: "Medium Veh",
    8: "Playground", 9: "Roundabout", 10: "Ship", 11: "Small Veh",
    12: "Pool", 13: "Tennis", 14: "Train"
}

# Couleurs fixes pour chaque classe (B, G, R) pour que ce soit lisible
COLORS = {
    1: (0, 255, 0),   # Basket = Vert fluo
    9: (0, 255, 255), # Rond-point = Jaune
    12: (255, 0, 255),# Piscine = Magenta
    # Les autres en rouge par défaut ou aléatoire
}

def visualize_all():
    # 1. Création du dossier de sortie
    if OUTPUT_DIR.exists():
        import shutil
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Lister les images
    print(f">>> Scan des images dans {IMG_DIR}...")
    if not IMG_DIR.exists():
        print("❌ Dossier images introuvable !")
        return

    img_files = list(IMG_DIR.glob("*.jpg")) + list(IMG_DIR.glob("*.jpeg"))
    print(f"   {len(img_files)} images trouvées.")

    count_processed = 0

    for img_path in tqdm(img_files):
        # Trouver le label associé
        lbl_path = LBL_DIR / (img_path.stem + ".txt")
        
        if not lbl_path.exists():
            continue # On saute les images sans label

        # Charger l'image
        img = cv2.imread(str(img_path))
        if img is None: continue
        h_img, w_img, _ = img.shape

        # Lire les annotations
        with open(lbl_path, 'r') as f:
            lines = f.readlines()

        has_objects = False
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5: continue
            
            cls_id = int(parts[0])
            xc, yc, w, h = map(float, parts[1:])

            # Conversion YOLO -> Pixels
            x1 = int((xc - w / 2) * w_img)
            y1 = int((yc - h / 2) * h_img)
            x2 = int((xc + w / 2) * w_img)
            y2 = int((yc + h / 2) * h_img)

            # Couleur (Spécifique ou Rouge par défaut)
            color = COLORS.get(cls_id, (0, 0, 255)) 
            
            # Dessiner le rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Ajouter le texte (Fond noir pour lisibilité)
            label = f"{cls_id}: {CLASS_NAMES.get(cls_id, '')}"
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - text_h - 5), (x1 + text_w, y1), color, -1) # Fond coloré
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
            
            has_objects = True

        # Sauvegarder l'image de debug
        if has_objects:
            cv2.imwrite(str(OUTPUT_DIR / img_path.name), img)
            count_processed += 1

    print(f"\n✅ Terminé ! {count_processed} images de vérification générées.")
    print(f"📂 Dossier : {OUTPUT_DIR}")
    print("👉 Zippe ce dossier et télécharge-le pour vérifier tes boîtes !")

if __name__ == "__main__":
    visualize_all()