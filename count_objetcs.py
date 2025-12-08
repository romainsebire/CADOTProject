import os
from pathlib import Path
from collections import Counter

# --- CONFIGURATION ---
# Vérifie bien que ce chemin est correct sur ton PC/Docker
LABEL_DIR = Path("CADOT_YOLO/train_original/labels")

# Noms des classes
CLASS_NAMES = {
  0: "--- IGNORED ---",
  1: "basketball field",
  2: "building",
  3: "crosswalk",
  4: "football field",
  5: "graveyard",
  6: "large vehicle",
  7: "medium vehicle",
  8: "playground",
  9: "roundabout",
  10: "ship",
  11: "small vehicle",
  12: "swimming pool",
  13: "tennis court",
  14: "train"
}

def count_instances():
    print(f">>> Analyse des fichiers dans {LABEL_DIR}...")
    
    if not LABEL_DIR.exists():
        print(f"ERREUR : Le dossier {LABEL_DIR} n'existe pas !")
        return

    stats = Counter()
    files = list(LABEL_DIR.glob("*.txt"))
    
    for txt_file in files:
        with open(txt_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if not line.strip(): continue
                try:
                    class_id = int(line.split()[0])
                    stats[class_id] += 1
                except ValueError:
                    pass
    
    # Calcul du total global
    total_objects = sum(stats.values())
    
    print(f"\n--- STATISTIQUES DU DATASET TRAIN (Total: {total_objects} objets) ---")
    print(f"{'ID':<4} | {'Nom de la classe':<20} | {'Nb':<8} | {'%':<8} | {'État'}")
    print("-" * 70)
    
    # Tri et ajout des classes vides
    sorted_stats = sorted(stats.items(), key=lambda item: item[1])
    all_ids = set(CLASS_NAMES.keys())
    found_ids = set(stats.keys())
    missing_ids = all_ids - found_ids
    for mid in missing_ids:
        sorted_stats.insert(0, (mid, 0))

    rare_threshold = 200
    
    for cls_id, count in sorted_stats:
        name = CLASS_NAMES.get(cls_id, "Unknown")
        
        # Calcul du pourcentage
        pct = (count / total_objects * 100) if total_objects > 0 else 0
        
        status = "CRITIQUE" if count < 50 else ("RARE" if count < rare_threshold else "OK")
        
        # Affichage formaté
        print(f"{cls_id:<4} | {name:<20} | {count:<8} | {pct:<7.2f}% | {status}")

if __name__ == "__main__":
    count_instances()