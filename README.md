# CADOT Project — Object Detection on Aerial Imagery of Paris

> **IEEE ICIP 2025 Grand Challenge** — Cityscape Aerial image Dataset for Object Detection  
> Academic project exploring data augmentation strategies to improve YOLOv11 detection of rare object classes in dense urban aerial imagery.

**Team members:** Romain Sebire · Pauline Rougeot · Rémy Plastre  
**Course:** Computer Vision — M2 2024/2025

---

## Project Overview

This project was developed as part of the [CADOT Challenge (IEEE ICIP 2025)](https://cadot.onrender.com/), which tasks participants with detecting 14 categories of urban objects in high-resolution aerial images of the Paris region, captured by the French National Institute of Geographic and Forest Information (IGN).

The core challenge is the **extreme class imbalance**: common classes like *small vehicle* (46.7%) and *building* (40.9%) dominate, while rare classes like *basketball field*, *football field*, *train*, or *swimming pool* each represent less than 1% of annotations. Standard training leads to models that excel on common classes but fail on rare ones.

**Our approach** combines three complementary strategies:
1. **Targeted Albumentations** — class-aware offline data augmentation with physics-based pipelines tailored to each object type
2. **Generative AI Inpainting** — synthetic data generation for the rarest class (basketball fields) using Google Gemini's built-in image generator (Imagen 3 / Nano Banana)
3. **YOLOv11 fine-tuning** — systematic comparison of model sizes and augmentation strategies

### Key Results

| Experiment | Model | Epochs | mAP50 | mAP50-95 | Precision | Recall |
|---|---|---|---|---|---|---|
| Baseline (no augmentation) | YOLOv11m | 81 | 0.529 | 0.364 | 0.524 | 0.526 |
| Classic Albumentations | YOLOv11m | 61 | 0.540 | 0.359 | 0.584 | 0.525 |
| **Multi-Pipeline Albumentations** | **YOLOv11m** | **85** | **0.558** | **0.373** | **0.560** | **0.568** |
| Baseline + GenAI Basketball | YOLOv11m | 70 | 0.530 | 0.363 | 0.522 | 0.562 |
| Baseline (smaller model) | YOLOv11n | 206 | 0.526 | 0.345 | 0.535 | 0.511 |

**Best model: Multi-Pipeline Albumentations — mAP50 = 0.558 (+5.5% over baseline)**

For reference, the official challenge baselines on the validation set are: YOLOv11 mAP50 = 62, YOLOv12 = 56, Faster R-CNN = 33.88, DiffusionDet = 52.76 (see [baseline comparison](docs/cadot_challenge_baseline_performance.png)).

---

## Dataset

- **Source:** [CADOT Challenge Page](https://cadot.onrender.com/dataset)
- **Download:** [CADOT_Dataset.zip](https://www-l2ti.univ-paris13.fr/iriser/dashboard/pages/download_CADOT_Dataset.php)
- **Format:** COCO (JSON annotations) — converted to YOLO format via our script
- **Images:** 4,628 high-resolution (500×500 px) aerial images from the Paris region
- **Annotations:** 106,691 bounding boxes across 14 categories
- **Split:** Train / Validation / Test (test labels withheld by organizers)

**14 Object Categories:**
basketball field, building, crosswalk, football field, graveyard, large vehicle, medium vehicle, playground, roundabout, ship, small vehicle, swimming pool, tennis court, train

---

## Installation

```bash
git clone https://github.com/romainsebire/CADOTProject.git
cd CADOTProject
pip install -r requirements.txt
```

---

## Reproducing Results

### Step 0 — Data Preparation

1. Download the dataset from the link above and extract `CADOT_Dataset/` into the project root.
2. Convert COCO annotations to YOLO format:

```bash
python convert_coco_to_yolo.py
```

This creates a `Dataset_YOLO/` folder with the proper directory structure (`train/images`, `train/labels`, `val/images`, `val/labels`).

3. Update `cadot.yaml` with the absolute path to your `Dataset_YOLO/` folder.

### Step 1 — Data Augmentation (choose one)

#### Option A: Classic Augmentation

Applies uniform transformations (rotation, flip, blur, brightness, CLAHE) to all images containing rare classes, multiplied by a fixed factor (default: ×5).

```bash
python augmentation_classic.py
```

Edit `RARE_CLASSES` and `AUGMENT_FACTOR` in the script to adjust.

#### Option B: Multi-Pipeline Augmentation (recommended)

Applies **class-specific pipelines** based on the physical properties of each object type:

| Pipeline | Target Classes | Strategy |
|---|---|---|
| **Sport** | Basketball, Football, Tennis | Preserves court lines and colors — only geometric transforms + brightness |
| **Texture** | Swimming Pool, Graveyard | Enhances fine details (waves, stones) — sharpening + CLAHE, no blur |
| **Shape** | Roundabout, Playground, Train, Ship | Allows geometric distortion and mild blur for shape-flexible objects |

Each class has an independent augmentation factor proportional to its rarity (e.g., basketball ×35, tennis ×5).

```bash
python augmentation_multi_pipeline.py
```

### Step 2 — Synthetic Data via Inpainting (optional)

This pipeline generates synthetic basketball courts in empty image regions using Generative AI. We used [Google Gemini's image generator](https://deepmind.google/models/gemini-image/) (Imagen 3, now rebranded as [Nano Banana](https://deepmind.google/models/gemini-image/)) to produce high-quality synthetic aerial views.

> **Why only basketball fields?** Our training infrastructure (Docker with NVIDIA T4) could only run Stable Diffusion 1.5 locally, which produced low-quality aerial imagery. Google Gemini's Imagen 3 delivered far superior results, but the semi-automated workflow (prepare masks locally → generate via Gemini → download → resize → verify) limited us to a single class demonstration.

```bash
# 1. Prepare masks and coordinates for empty regions
python prepare_inpainting.py

# 2. Upload INPAINTING_STAGING/images/ and masks/ to your generation service
#    Prompt: "Satellite view of a basketball court, distinct white lines,
#    asphalt surface, top-down orthographic view, high resolution"

# 3. Place generated images in GENERATED_RESULTS/
#    Ensure filenames match the originals from step 1

# 4. Resize generated images to 500x500
python resize_image.py

# 5. Merge into dataset with automatic labeling
python merge_results.py

# 6. Visual verification of bounding box alignment
python visualize_bbox.py
```

### Step 3 — Training

```bash
python train.py
```

Key parameters (edit in script):
- `model`: `yolo11n.pt` (Nano, 2.6M params) or `yolo11m.pt` (Medium, 20.1M params)
- `epochs`: 300 (with early stopping, patience=20)
- `batch`: 16
- `imgsz`: 512 (matching 500×500 source images)
- `name`: output folder name under `runs/`
- `device`: `0` for CUDA GPU, `mps` for Apple Silicon

Our models were trained on a **Docker container with an NVIDIA T4 GPU** (16 GB VRAM).

### Step 4 — Evaluation

Training results (metrics, curves, confusion matrices) are automatically saved under `runs/<name>/`. Key files:
- `results.csv` — per-epoch metrics
- `results.png` — training curves
- `confusion_matrix.png` — class-level predictions
- `BoxF1_curve.png`, `BoxPR_curve.png` — F1 and Precision-Recall curves

---

## Project Structure

```
CADOTProject/
├── train.py                        # YOLOv11 training script
├── convert_coco_to_yolo.py         # COCO → YOLO format conversion
├── augmentation_classic.py         # Uniform augmentation for rare classes
├── augmentation_multi_pipeline.py  # Class-specific augmentation pipelines
├── prepare_inpainting.py           # Mask generation for synthetic data
├── resize_image.py                 # Resize generated images to 500x500
├── merge_results.py                # Merge synthetic images into dataset
├── visualize_bbox.py               # Visual debug of bounding boxes
├── count_objects.py                # Dataset class distribution analysis
├── cadot.yaml                      # YOLO dataset configuration
├── requirements.txt                # Python dependencies
├── docs/
│   ├── cadot_challenge_baseline_performance.png
│   └── references/                 # YOLO research papers
├── samples/                        # Sample images (10 per folder)
│   ├── yolo/                       # YOLO-format dataset samples
│   │   ├── train/images/ & labels/
│   │   └── val/images/ & labels/
│   └── augmented/                  # Augmentation output samples
│       ├── images/ & labels/
│       └── visual_debug/           # Bounding box verification
├── runs/                           # Training results (metrics, curves, plots)
│   ├── finetune_v11n/
│   ├── finetune_v11m/
│   ├── finetune_v11m_albumentations_classique/
│   ├── finetune_v11m_albumentations_multipipelines/  ← Best model (weights included)
│   └── finetune_v11m_iagenbasketball/
└── INPAINTING_STAGING/             # Generated masks and coordinates (15 examples)
```

> **Note:** The full dataset is not included in this repository. Download it from the [CADOT Challenge page](https://www-l2ti.univ-paris13.fr/iriser/dashboard/pages/download_CADOT_Dataset.php). The `samples/` folder contains 10 representative images per split for reference.

---

## Detailed Analysis of Results

### Multi-Pipeline Augmentation vs Baseline

The multi-pipeline approach yielded the best overall improvement (+5.5% mAP50), demonstrating that **class-aware augmentation** outperforms both no augmentation and uniform augmentation. The key insight is that different object types have different invariance properties:
- Sports courts must preserve their **line markings and colors** (rotation is fine, blur is not)
- Textured objects like pools and graveyards benefit from **contrast enhancement** but not geometric distortion
- Shape-flexible objects like roundabouts tolerate **elastic transforms** that would destroy court lines

### GenAI Inpainting for Basketball Fields

The inpainting experiment specifically targeted the *basketball field* class, which has **0 mAP50 on both val and test** in the Faster R-CNN baseline and only 52% with YOLOv11. While the overall mAP50 improvement was modest (+0.1%), the recall increased from 0.526 to 0.562 (+6.8%), suggesting the model became better at finding objects it previously missed. A larger-scale generation effort covering all rare classes would likely yield stronger gains, but was limited by our infrastructure constraints (see Step 2 above).

### Model Size Comparison

YOLOv11n (Nano) trained for 206 epochs before early stopping and reached mAP50 = 0.526, comparable to the YOLOv11m baseline (0.529) but with significantly fewer parameters. The Medium model benefited more from augmentation, suggesting that larger models can better leverage additional training data.

---

## Model Weights

To keep the repository size manageable, model weights (`.pt`) are only included for the **best performing model**:

> `runs/finetune_v11m_albumentations_multipipelines/weights/best.pt`

Use this file for inference or further fine-tuning.

---

## References

- [CADOT Challenge (IEEE ICIP 2025)](https://cadot.onrender.com/)
- [Ultralytics YOLOv11 Documentation](https://docs.ultralytics.com/)
- [Albumentations Library](https://albumentations.ai/)
- Research papers in [`docs/references/`](docs/references/)

---

## License

This project was developed for academic purposes as part of the CADOT Challenge. The dataset is provided by [LabCom IRISER](https://www-l2ti.univ-paris13.fr/iriser/) (ANR-21-LCV3-0004).
