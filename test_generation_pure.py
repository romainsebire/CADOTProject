import torch
from diffusers import AutoPipelineForText2Image, AutoencoderKL
from PIL import Image

# Modèle de base
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
OUTPUT_IMAGE = "test_court_pure.png"

def test_pure_generation():
    print(f">>> Chargement de SDXL avec VAE optimisé fp16...")
    
    try:
        # 1. On charge un VAE "magique" corrigé pour le fp16
        # Cela évite les erreurs de type ET les images noires, tout en économisant la RAM
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix", 
            torch_dtype=torch.float16
        )

        # 2. On charge le pipeline en lui injectant ce VAE
        pipe = AutoPipelineForText2Image.from_pretrained(
            MODEL_ID,
            vae=vae,       # <--- On remplace le VAE par défaut ici
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        ).to("cuda")
        
        pipe.enable_attention_slicing()
        
    except Exception as e:
        print(f"Erreur chargement : {e}")
        return

    # Prompt
    prompt = "A full basketball court seen from straight above, satellite view, bright sunny day, asphalt surface, distinct white painted lines, isolated on a neutral grey background"
    negative_prompt = "shadows, buildings, trees, perspective, isometric, 3d render, blurry, dark"

    print(f">>> Génération en cours...")
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=25,
        guidance_scale=9.0,
        height=768, 
        width=768
    ).images[0]
    
    image.save(OUTPUT_IMAGE)
    print(f"Image générée : {OUTPUT_IMAGE}")

if __name__ == "__main__":
    test_pure_generation()