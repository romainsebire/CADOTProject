from PIL import Image
import random
from pathlib import Path
import os

# --- CONFIGURATION ---
IMG_DIR = Path("CADOT_YOLO/train_original/images")
OUTPUT_DIR = Path("debug_collage")
FOREGROUND_IMG = "test_court_pure.png" # L'image générée à l'étape 1

def test_collage():
    print(">>> Démarrage du test de collage...")
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # 1. Charger le terrain de basket généré (Foreground)
    if not os.path.exists(FOREGROUND_IMG):
        print(f"❌ Erreur : Génère d'abord {FOREGROUND_IMG} avec l'étape 1 !")
        return
    fg_image_full = Image.open(FOREGROUND_IMG).convert("RGBA")

    # 2. Charger une image de fond au hasard (Background)
    background_files = list(IMG_DIR.glob("*.jpg"))
    if not background_files:
        print("❌ Erreur : Pas d'images de fond trouvées.")
        return
    bg_path = background_files[0] # On prend la première pour le test
    print(f"   Fond utilisé : {bg_path}")
    bg_image = Image.open(bg_path).convert("RGBA")
    w_bg, h_bg = bg_image.size

    # 3. Redimensionner le terrain de basket
    # On veut qu'il fasse environ 150x100 pixels
    target_w, target_h = 150, 100
    fg_resized = fg_image_full.resize((target_w, target_h), Image.Resampling.LANCZOS)

    # 4. Choisir une position pour coller (Ex: coin haut gauche, avec une marge)
    x_pos = 100
    y_pos = 100
    print(f"   Position du collage : ({x_pos}, {y_pos})")

    # 5. LE COLLAGE (Paste)
    # On copie le fond pour ne pas modifier l'original
    final_composition = bg_image.copy()
    # On colle le premier plan (fg) sur le fond à la position (x,y)
    # Le 3ème argument sert de masque de transparence si le PNG en a un
    final_composition.paste(fg_resized, (x_pos, y_pos), fg_resized)

    # 6. Sauvegarde
    save_path = OUTPUT_DIR / "collage_result.png" # PNG pour garder la qualité
    final_composition.save(save_path)
    print(f"✅ Collage terminé ! Résultat : {save_path}")

    # 7. Afficher les coordonnées YOLO pour vérification
    xc = (x_pos + target_w / 2) / w_bg
    yc = (y_pos + target_h / 2) / h_bg
    wn = target_w / w_bg
    hn = target_h / h_bg
    print(f"\n--- Coordonnées YOLO théoriques à ajouter au .txt ---")
    print(f"1 {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")

if __name__ == "__main__":
    test_collage()