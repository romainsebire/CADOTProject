import torch
from diffusers import StableDiffusionXLInpaintPipeline
from PIL import Image, ImageDraw
import numpy as np
import os
from pathlib import Path
import random
from tqdm import tqdm

# --- CONFIGURATION ---
IMG_DIR = Path("CADOT_YOLO/train_original/images")
LBL_DIR = Path("CADOT_YOLO/train_original/labels")
OUTPUT_IMG_DIR = Path("generated_images/raw_model/images") 
OUTPUT_LBL_DIR = Path("generated_images/raw_model/labels")

# Modèle SDXL Inpainting (Le top pour un T4)
MODEL_ID = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"

# Tâches (Terrains de Basket)
GENERATION_TASKS = {
    1: { # Basketball
        # Prompt SDXL : Il comprend mieux le langage naturel
        "prompt": "Aerial view of a basketball court, distinct painted white lines, orange or blue surface, asphalt texture, outdoor, top-down orthographic satellite view",
        "size_range": (80, 150), # Un peu plus grand pour que SDXL ait de la place
        "count": 1 # Test avec 10 pour commencer
    }
}

def generate_synthetic_data():
    print(f">>> Chargement de SDXL Inpainting ({MODEL_ID})...")
    
    # On charge en Float16 pour tenir sur le T4 (sinon ça plante)
    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    )
    pipe.to("cuda")
    
    # Optimisation mémoire indispensable sur T4 pour SDXL
    pipe.enable_attention_slicing()
    # Si ça plante encore en mémoire, décommente la ligne suivante :
    # pipe.enable_model_cpu_offload()

    background_files = list(IMG_DIR.glob("*.jpg"))

    for cls_id, params in GENERATION_TASKS.items():
        print(f"\n--- Génération Classe {cls_id} ---")
        
        for i in tqdm(range(params['count'])):
            # 1. Image de fond
            bg_path = random.choice(background_files)
            image_source = Image.open(bg_path).convert("RGB")
            w_img, h_img = image_source.size

            # 2. Zone aléatoire
            obj_w = random.randint(params['size_range'][0], params['size_range'][1])
            obj_h = int(obj_w * random.uniform(0.6, 1.4)) # Ratio varié
            
            if w_img - obj_w < 10 or h_img - obj_h < 10: continue 
            
            x1 = random.randint(0, w_img - obj_w)
            y1 = random.randint(0, h_img - obj_h)
            x2, y2 = x1 + obj_w, y1 + obj_h

            # 3. Masque
            mask_image = Image.new("L", (w_img, h_img), 0)
            draw = ImageDraw.Draw(mask_image)
            draw.rectangle([(x1, y1), (x2, y2)], fill=255)

            # 4. Génération SDXL
            # On force la "strength" à 1.0 (remplacement total) 
            # et guidance_scale élevé pour forcer l'objet
            generated_image = pipe(
                prompt=params["prompt"],
                negative_prompt="building, house, trees, blur, low resolution, 3d render, perspective, shadow",
                image=image_source,
                mask_image=mask_image,
                num_inference_steps=25, # Suffisant pour SDXL
                guidance_scale=12.0,    # TRES FORT pour forcer l'apparition du terrain
                strength=0.99           # On remplace presque tout dans le masque
            ).images[0]

            # 5. Sauvegarde
            new_filename = f"sdxl_{cls_id}_{i}_{bg_path.stem}"
            save_img_path = OUTPUT_IMG_DIR / (new_filename + ".jpg")
            save_lbl_path = OUTPUT_LBL_DIR / (new_filename + ".txt")
            
            generated_image.save(save_img_path)

            # 6. Label (Copie ancien + Ajout nouveau)
            existing_labels = []
            original_lbl_path = LBL_DIR / (bg_path.stem + ".txt")
            if original_lbl_path.exists():
                with open(original_lbl_path, 'r') as f: existing_labels = f.readlines()

            with open(save_lbl_path, 'w') as f_out:
                for line in existing_labels: f_out.write(line)
                
                # Ajout du label SDXL
                xc, yc = (x1 + obj_w/2)/w_img, (y1 + obj_h/2)/h_img
                wn, hn = obj_w/w_img, obj_h/h_img
                f_out.write(f"{cls_id} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}\n")

if __name__ == "__main__":
    generate_synthetic_data()