import cv2
import os
from pathlib import Path
from tqdm import tqdm

# --- CONFIGURATION ---
GENERATED_DIR = Path("GENERATED_RESULTS")
TARGET_SIZE = (500, 500) # La taille originale de tes images IGN

def resize_images():
    print(f">>> Redimensionnement des images dans {GENERATED_DIR} vers {TARGET_SIZE}...")
    
    if not GENERATED_DIR.exists():
        print(f"❌ Erreur : Le dossier {GENERATED_DIR} n'existe pas.")
        return

    # On cherche jpg, jpeg, png
    files = list(GENERATED_DIR.glob("*.jpg")) + list(GENERATED_DIR.glob("*.png")) + list(GENERATED_DIR.glob("*.jpeg"))
    
    if not files:
        print("⚠️ Aucune image trouvée.")
        return

    count = 0
    for img_path in tqdm(files):
        # 1. Lire l'image
        img = cv2.imread(str(img_path))
        if img is None: continue
        
        h, w, _ = img.shape
        
        # Si l'image n'est pas déjà en 500x500
        if (w, h) != TARGET_SIZE:
            # 2. Redimensionner (INTER_AREA est le meilleur algo pour réduire la taille)
            resized_img = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)
            
            # 3. Écraser l'ancien fichier
            # Si c'était un PNG, on peut le convertir en JPG pour uniformiser le dataset
            new_path = img_path.with_suffix('.jpg')
            cv2.imwrite(str(new_path), resized_img)
            
            # Si l'extension a changé (ex: png -> jpg), on supprime l'ancien
            if new_path != img_path:
                os.remove(img_path)
                
            count += 1

    print(f"\n✅ Terminé ! {count} images ont été redimensionnées en 500x500.")
    print("Tu peux maintenant lancer 'merge_results.py'.")

if __name__ == "__main__":
    resize_images()