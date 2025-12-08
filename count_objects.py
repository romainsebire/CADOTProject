import os
from pathlib import Path
from collections import Counter

# --- CONFIGURATION ---
LABEL_DIR = Path("Dataset_YOLO/train/labels")

# Class names
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
    print(f">>> Analyzing files in {LABEL_DIR}...")
    
    if not LABEL_DIR.exists():
        print(f"ERROR: The directory {LABEL_DIR} does not exist!")
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
    
    # Calculate global total
    total_objects = sum(stats.values())
    
    print(f"\n--- TRAIN DATASET STATISTICS (Total: {total_objects} objects) ---")
    print(f"{'ID':<4} | {'Class Name':<20} | {'Count':<8} | {'%':<8} | {'Status'}")
    print("-" * 70)
    
    # Sort and add empty classes
    sorted_stats = sorted(stats.items(), key=lambda item: item[1])
    all_ids = set(CLASS_NAMES.keys())
    found_ids = set(stats.keys())
    missing_ids = all_ids - found_ids
    for mid in missing_ids:
        sorted_stats.insert(0, (mid, 0))

    rare_threshold = 200
    
    for cls_id, count in sorted_stats:
        name = CLASS_NAMES.get(cls_id, "Unknown")
        
        # Calculate percentage
        pct = (count / total_objects * 100) if total_objects > 0 else 0
        
        status = "CRITICAL" if count < 50 else ("RARE" if count < rare_threshold else "OK")
        
        # Formatted output
        print(f"{cls_id:<4} | {name:<20} | {count:<8} | {pct:<7.2f}% | {status}")

if __name__ == "__main__":
    count_instances()