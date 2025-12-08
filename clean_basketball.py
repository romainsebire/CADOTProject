import os
from pathlib import Path
from tqdm import tqdm

# --- CONFIGURATION ---
# Le dossier "sale" (celui avec les x35 baskets)
IMG_DIR = Path("CADOT_YOLO/train/images")
LBL_DIR = Path("CADOT_YOLO/train/labels")

# L'ID de la classe à nettoyer (Basketball)
TARGET_CLASS = 1 

def clean_specific_class_augmentations():
    print(f">>> Nettoyage des augmentations de la classe {TARGET_CLASS} (Basketball)...")
    
    # 1. Lister tous les fichiers générés par le script expert
    # Ils se terminent par "_expert_X.txt"
    aug_files = list(LBL_DIR.glob("*_expert_*.txt"))
    
    deleted_count = 0
    
    for lbl_path in tqdm(aug_files):
        # 2. Vérifier si ce fichier contient du Basket
        has_basket = False
        with open(lbl_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith(f"{TARGET_CLASS} "):
                    has_basket = True
                    break
        
        # 3. Si c'est une image augmentée contenant du basket -> On supprime !
        if has_basket:
            # Suppression du label
            os.remove(lbl_path)
            
            # Suppression de l'image associée (.jpg)
            img_name = lbl_path.stem + ".jpg"
            img_path = IMG_DIR / img_name
            if img_path.exists():
                os.remove(img_path)
            
            deleted_count += 1

    print(f"\n✅ Nettoyage terminé !")
    print(f"🗑️ {deleted_count} images augmentées de Basket ont été supprimées.")
    print("ℹ️ Les 15 images originales et les augmentations des autres classes sont conservées.")

if __name__ == "__main__":
    clean_specific_class_augmentations()