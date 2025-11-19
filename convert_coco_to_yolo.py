from pycocotools.coco import COCO
from pathlib import Path
import shutil
import os
import json

# ====== À ADAPTER ======
BASE = Path("/Users/remyplastre/Downloads/CADOT_Dataset")  # dossier qui contient train, valid, test
# Par ex. si tu as : /home/remy/cadot_dataset/train/annotations.json
# mets BASE = Path("/home/remy/cadot_dataset")
# =======================

splits = ["train", "valid", "test"]

# --- 1) Construire un mapping global COCO category_id -> index YOLO ---
train_ann = BASE / "train" / "_annotations.coco.json"
coco_train = COCO(str(train_ann))

cats = coco_train.loadCats(coco_train.getCatIds())
# on trie par id COCO pour avoir un ordre stable
cats_sorted = sorted(cats, key=lambda c: c["id"])
id_map = {c["id"]: i for i, c in enumerate(cats_sorted)}  # ex: {3:0, 5:1, ...}
names = [c["name"] for c in cats_sorted]

print("Mapping catégories (COCO id -> YOLO id):")
for old_id, new_id in id_map.items():
    print(old_id, "→", new_id)
print("Noms de classes:", names)

# --- 2) Conversion pour chaque split ---
for split in splits:
    print(f"\n=== Traitement du split: {split} ===")
    ann_file = BASE / split / "_annotations.coco.json"
    
    # Vérifier si le fichier d'annotations existe
    if not ann_file.exists():
        print(f"⚠️  Fichier d'annotations introuvable pour '{split}': {ann_file}")
        print(f"    Le split '{split}' sera ignoré.")
        continue
    
    coco = COCO(str(ann_file))

    labels_dir = BASE / "yolo" / "labels" / split
    images_out_dir = BASE / "yolo" / "images" / split
    labels_dir.mkdir(parents=True, exist_ok=True)
    images_out_dir.mkdir(parents=True, exist_ok=True)

    # Si tes images sont dans BASE/split/images, adapte cette variable
    # Exemple 1: images directement dans BASE/split
    img_root = BASE / split
    # Exemple 2: images dans BASE/split/images
    # img_root = BASE / split / "images"

    for img_id, img_info in coco.imgs.items():
        file_name = img_info["file_name"]
        width = img_info["width"]
        height = img_info["height"]

        # --- Copier l'image dans la nouvelle structure YOLO ---
        src_img = img_root / file_name
        dst_img = images_out_dir / file_name
        dst_img.parent.mkdir(parents=True, exist_ok=True)

        if not dst_img.exists():
            if not src_img.exists():
                print(f"[WARN] Image introuvable: {src_img}")
                continue
            shutil.copy2(src_img, dst_img)

        # --- Créer le fichier .txt YOLO associé ---
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        if len(anns) == 0:
            # image sans annotations : on peut créer un .txt vide
            # ou passer; YOLO accepte les deux
            open(labels_dir / (Path(file_name).stem + ".txt"), "w").close()
            continue

        yolo_label_path = labels_dir / (Path(file_name).stem + ".txt")
        with open(yolo_label_path, "w") as f:
            for ann in anns:
                if ann.get("iscrowd", 0) == 1:
                    continue

                x, y, w, h = ann["bbox"]  # COCO: (x_min, y_min, width, height)

                # Conversion COCO -> YOLO (coordonnées normalisées)
                cx = (x + w / 2) / width
                cy = (y + h / 2) / height
                nw = w / width
                nh = h / height

                coco_cat_id = ann["category_id"]
                cls_id = id_map[coco_cat_id]  # 0..nc-1

                f.write(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

print("\nConversion terminée ✅")

# --- 3) Écriture d'un data.yaml de base ---
data_yaml = {
    "path": str((BASE / "yolo").resolve()),
    "train": "images/train",
    "val": "images/valid",
    "test": "images/test",
    "nc": len(names),
    "names": names,
}

yaml_path = BASE / "yolo" / "data.yaml"
try:
    import yaml
    with open(yaml_path, "w") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False, allow_unicode=True)
    print(f"Fichier data.yaml écrit dans: {yaml_path}")
except ImportError:
    print("\n[INFO] Le module pyyaml n'est pas installé.")
    print("Installe-le avec: pip install pyyaml")
    print("Puis crée un data.yaml avec le contenu suivant :")
    print(data_yaml)
