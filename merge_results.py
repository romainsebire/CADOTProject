import shutil
import os
from pathlib import Path

# --- CONFIGURATION ---
GENERATED_DIR = Path("GENERATED_RESULTS") # Là où tu as mis tes images finies
COORD_DIR = Path("INPAINTING_STAGING/coords") # Là où le script 1 a mis les infos

FINAL_IMG_DIR = Path("CADOT_YOLO/data_augmentation/images")
FINAL_LBL_DIR = Path("CADOT_YOLO/data_augmentation/labels")
ORIGINAL_LBL_DIR = Path("CADOT_YOLO/train_original/labels") # Pour récupérer les labels du fond

def merge_data():
    print(">>> Fusion des images générées dans le dataset...")
    
    gen_files = list(GENERATED_DIR.glob("*.jpg")) # Ou .png selon ton outil
    
    count = 0
    for gen_img_path in gen_files:
        # Le nom de fichier doit correspondre à celui généré à l'étape 1
        # ex: basket_gen_0_image123.jpg
        # Attention si ton outil rename les fichiers, renomme-les avant !
        stem = gen_img_path.stem
        
        # 1. Retrouver les coordonnées sauvegardées
        coord_path = COORD_DIR / (stem + ".txt")
        if not coord_path.exists():
            print(f"⚠️ Pas de coordonnées trouvées pour {stem}, ignoré.")
            continue
            
        # 2. Retrouver les labels originaux (les autres objets de l'image de fond)
        # On extrait le nom de l'image originale depuis le nom composé
        # basket_gen_0_NOMORIGINAL
        original_stem = stem.split('_', 3)[3] 
        original_lbl_path = ORIGINAL_LBL_DIR / (original_stem + ".txt")
        
        existing_labels = []
        if original_lbl_path.exists():
            with open(original_lbl_path, 'r') as f:
                existing_labels = f.readlines()
        
        # 3. Créer le nouveau fichier label complet
        final_lbl_path = FINAL_LBL_DIR / (stem + ".txt")
        
        with open(final_lbl_path, 'w') as f_out:
            # A. Les anciens objets
            for line in existing_labels:
                f_out.write(line)
            
            # B. Le nouveau terrain de basket
            with open(coord_path, 'r') as f_coord:
                new_obj_line = f_coord.read()
                f_out.write(new_obj_line)
        
        # 4. Copier l'image générée dans le dossier Train
        shutil.copy(gen_img_path, FINAL_IMG_DIR / (stem + ".jpg"))
        
        count += 1

    print(f"\n✅ Terminé ! {count} nouvelles images synthétiques ajoutées au dataset.")
    print("Tu peux maintenant lancer ton script Albumentations 'Expert' pour les multiplier !")

if __name__ == "__main__":
    merge_data()