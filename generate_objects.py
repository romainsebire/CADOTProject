import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image, ImageDraw
import numpy as np
import os
from pathlib import Path
import random
from tqdm import tqdm

# --- CONFIGURATION ---
IMG_DIR = Path("CADOT_YOLO/train_original/images")
LBL_DIR = Path("CADOT_YOLO/train_original/labels")
OUTPUT_IMG_DIR = Path("CADOT_YOLO_FINAL/train/images")
OUTPUT_LBL_DIR = Path("CADOT_YOLO_FINAL/train/labels")

# Modèle (Optimisé pour GPU T4)
MODEL_ID = "stabilityai/stable-diffusion-2-inpainting"

# Quoi générer ? (Classe ID : Prompt)
GENERATION_TASKS = {
    1: { # Basketball field
        # Prompt optimisé pour la vue satellite (surface dure, lignes, extérieur)
        "prompt": "satellite view of an outdoor basketball court, asphalt or concrete surface with painted white lines, urban park or school context, aerial photography, top down orthographic view, high resolution",
        "size_range": (60, 120), 
        # Classe très critique (seulement 15 objets réels), on en génère 50
        "count": 10 
    },
    #12: { # Swimming pool
    #    "prompt": "satellite view of a rectangular swimming pool, blue water, backyard context, aerial photography, high resolution, orthographic view",
    #    "size_range": (30, 60), # Taille en pixels (approx)
    #    "count": 50 # Combien en créer ?
    #},
    #9: { # Roundabout
    #    "prompt": "satellite view of a asphalt roundabout, road intersection, aerial photography, top down view, realistic texture",
    #    "size_range": (80, 150),
    #    "count": 40
    #},
    #4: { # Football field
    #    "prompt": "satellite view of a green football field, white lines, grass texture, aerial photography, top down",
    #    "size_range": (100, 200),
    #    "count": 30
    #}
}

def generate_synthetic_data():
    print(f">>> Chargement du modèle {MODEL_ID} sur GPU...")
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16, # Important pour économiser la VRAM du T4
    )
    pipe = pipe.to("cuda")
    pipe.enable_attention_slicing() # Optimisation mémoire

    # Liste des images disponibles pour servir de "fond"
    background_files = list(IMG_DIR.glob("*.jpg"))

    for cls_id, params in GENERATION_TASKS.items():
        print(f"\n--- Génération de la classe {cls_id} ({params['count']} images) ---")
        
        for i in tqdm(range(params['count'])):
            # 1. Choisir une image de fond au hasard
            bg_path = random.choice(background_files)
            image_source = Image.open(bg_path).convert("RGB")
            w_img, h_img = image_source.size

            # 2. Définir une zone aléatoire pour l'objet (La Bounding Box)
            # ATTENTION : Idéalement, il faudrait vérifier qu'on ne chevauche pas un autre objet
            # Ici on fait simple (random), mais risque de collision.
            obj_w = random.randint(params['size_range'][0], params['size_range'][1])
            obj_h = random.randint(params['size_range'][0], params['size_range'][1]) # Carré ou presque
            
            # Position aléatoire (en évitant les bords)
            if w_img - obj_w < 10 or h_img - obj_h < 10: continue 
            
            x1 = random.randint(0, w_img - obj_w)
            y1 = random.randint(0, h_img - obj_h)
            x2 = x1 + obj_w
            y2 = y1 + obj_h

            # 3. Créer le Masque (Blanc = Zone à modifier, Noir = Touche pas)
            mask_image = Image.new("L", (w_img, h_img), 0) # Tout noir
            draw = ImageDraw.Draw(mask_image)
            draw.rectangle([(x1, y1), (x2, y2)], fill=255) # Zone blanche

            # 4. Génération (Inpainting)
            # On utilise un "Negative Prompt" pour éviter les mauvaises vues
            generated_image = pipe(
                prompt=params["prompt"],
                negative_prompt="text, watermark, low quality, blurry, perspective, side view, trees covering, shadows",
                image=image_source,
                mask_image=mask_image,
                num_inference_steps=30, # 30 est un bon compromis qualité/vitesse
                guidance_scale=7.5
            ).images[0]

            # 5. Sauvegarde
            new_filename = f"synth_{cls_id}_{i}_{bg_path.stem}"
            save_img_path = OUTPUT_IMG_DIR / (new_filename + ".jpg")
            save_lbl_path = OUTPUT_LBL_DIR / (new_filename + ".txt")

            generated_image.save(save_img_path)

            # 6. Création du Label YOLO
            # Il faut reprendre les labels existants de l'image de fond et AJOUTER le nouveau
            existing_labels = []
            original_lbl_path = LBL_DIR / (bg_path.stem + ".txt")
            if original_lbl_path.exists():
                with open(original_lbl_path, 'r') as f:
                    existing_labels = f.readlines()

            with open(save_lbl_path, 'w') as f_out:
                # Réécrire les anciens labels
                for line in existing_labels:
                    f_out.write(line)
                
                # Ajouter le NOUVEAU label synthétique
                # Conversion YOLO (x_center, y_center, w, h) normalisé
                xc = (x1 + obj_w / 2) / w_img
                yc = (y1 + obj_h / 2) / h_img
                wn = obj_w / w_img
                hn = obj_h / h_img
                
                f_out.write(f"{cls_id} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}\n")

if __name__ == "__main__":
    generate_synthetic_data()