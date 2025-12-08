# CADOT YOLO Project - Object Detection on Aerial Imagery

This repository contains the complete workflow for training a YOLOv11 model to detect objects in aerial images (CADOT dataset). It includes pipelines for **Data Preprocessing**, **Generative AI Inpainting**, **Advanced Data Augmentation** (Albumentations), and **Training**.

---

**Before running any augmentation or inpainting script:**

The scripts below could modify the dataset or generate files that mimic the original structure. To prevent "pollution" of your clean dataset:

1.  **Duplicate** your original training folder.
2.  Keep a safe copy of `Dataset_YOLO/train_original`.
3.  Perform augmentations on a working copy (e.g., `Dataset_YOLO/train`), never on the backup.

---

## 1. Synthetic Data Generation (Inpainting)

We use **Google Nano Banana** to generate rare objects (e.g., Basketball courts) in empty areas of existing images.

### Step 1: Preparation
Run the script to identify empty spots and generate masks.
```bash
python prepare_inpainting.py
```
* **Output:** Creates a folder `INPAINTING_STAGING/` containing:
    * `images/`: Original crop context.
    * `masks/`: Black and white binary masks.
    * `coords/`: YOLO coordinates for the future object.

### Step 2: Cloud Generation (External)

Upload the content of `images/` and `masks/` to a Generative AI service (e.g., Nano Banana).

* **Prompt Model:**
    > "Satellite view of a basketball court, distinct white lines, asphalt surface, top-down orthographic view, high resolution, realistic lighting, integration with background."

### Step 3: File Management

1.  Download the generated images.
2.  **Renaming:** Ensure the generated filenames match the original filenames from Step 1 (e.g., `basket_gen_0_image123.jpg`).
3.  Place them in the `GENERATED_RESULTS/` folder.

### Step 4: Resize

The generator might output 1024x1024 images. Resize them back to 500x500 to match the mask coordinates.
```bash
python resize_generated.py
```

### Step 5: Merge & Labeling

Inject the synthetic images into the dataset and automatically create the labels.
```bash
python merge_results.py
```

* **Result:** New images are added to `Dataset_YOLO/data_augmentation/`.

### Step 6: Visual Verification

**Always** check if the generated object aligns with the computed bounding box.
```bash
python visualize_bbox.py
```

* Check the output in `Dataset_YOLO/data_augmentation/visual_debug`.

### Step 7: Final Integration
If visualizations are correct, **manually move** the verified images and labels from `Dataset_YOLO/data_augmentation/` to your main training folder:

* `cp Dataset_YOLO/data_augmentation/images/* -> Dataset_YOLO/train/images/`
* `cp Dataset_YOLO/data_augmentation/labels/* -> Dataset_YOLO/train/labels/`

---

## 2. Data Augmentation (Albumentations)

We use **Albumentations** to multiply the dataset. Choose **ONE** of the following strategies.

### Option A: Simple Augmentation (General)
Applies the same transformations (Flip, Blur, Brightness) to all rare classes.

* **Config:** Edit `albumentation_classique.py` to change `AUGMENT_FACTOR` (e.g., 5).
* **Run:**
```bash
python albumentation_classique.py
```

### Option B: Multi Pipelines Augmentation
Applies specific physics-based transformations depending on the object type (e.g., "Sport" pipeline preserves lines, "Shape" pipeline distorts roundabouts).

* **Config:** Edit `albumentation_pipelines.py`.
    * Adjust `augment_counts` dictionary to set target quantities per class.
* **Run:**
```bash
python albumentation_pipelines.py
```

---

## 3. Training the Model

Once the dataset is ready (Originals + Augmented), launch the YOLO training.

### Configuration
Ensure your `data.yaml` points to the correct train/val paths.

### Launch Command
```bash
python lancer_train.py
```
### Key Parameters:
* `model`: `yolo11n.pt` (Nano) or `yolo11m.pt` (Medium).
* `epochs`: Number of passes (300).
* `name`: **Important!** Name of the output folder (e.g., `finetune_v11m`).
* `imgsz`: Input image size (512 for this dataset).

---

## 4. Evaluation & Visualization

To check the results after training:

1.  **Metrics:** Look at the `results.csv` in `runs/name/`. Focus on **mAP50**.