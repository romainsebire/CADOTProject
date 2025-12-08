import cv2
import os
from pathlib import Path
from tqdm import tqdm

# --- CONFIGURATION ---
GENERATED_DIR = Path("GENERATED_RESULTS")
TARGET_SIZE = (500, 500) # The original size of the IGN images

def resize_images():
    print(f">>> Resizing images in {GENERATED_DIR} to {TARGET_SIZE}...")
    
    if not GENERATED_DIR.exists():
        print(f"Error: The directory {GENERATED_DIR} does not exist.")
        return

    # Looking for jpg, jpeg, png
    files = list(GENERATED_DIR.glob("*.jpg")) + list(GENERATED_DIR.glob("*.png")) + list(GENERATED_DIR.glob("*.jpeg"))
    
    if not files:
        print("No images found.")
        return

    count = 0
    for img_path in tqdm(files):
        # 1. Read the image
        img = cv2.imread(str(img_path))
        if img is None: continue
        
        h, w, _ = img.shape
        
        # If the image is not already 500x500
        if (w, h) != TARGET_SIZE:
            # 2. Resize (INTER_AREA is the best algo for downscaling)
            resized_img = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)
            
            # 3. Overwrite the old file
            # If it was a PNG, we convert to JPG to standardize the dataset
            new_path = img_path.with_suffix('.jpg')
            cv2.imwrite(str(new_path), resized_img)
            
            # If the extension changed (e.g. png -> jpg), remove the old one
            if new_path != img_path:
                os.remove(img_path)
                
            count += 1

    print(f"\nFinished! {count} images resized to 500x500.")
    print("You can now run 'merge_results.py'.")

if __name__ == "__main__":
    resize_images()