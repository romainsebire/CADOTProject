# run_full_training.py
import os
import math
import subprocess
import torch

TOTAL_EPOCHS = 100
BLOCK_SIZE = 2

RUN_NAME = "cadot_yolo11n_stable"
BASE_MODEL = "runs/detect/cadot_yolo11n_stable/weights/last.pt"
DATA = "CADOT_Dataset/yolo/data.yaml"


def get_last_epoch():
    checkpoint_path = os.path.join("runs", "detect", RUN_NAME, "weights", "last.pt")
    if not os.path.exists(checkpoint_path):
        print("📭 Aucun checkpoint → epoch = -1 (début)")
        return -1
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    last_epoch = ckpt.get("epoch", -1)
    print(f"📌 Checkpoint trouvé → epoch actuel = {last_epoch}")
    return last_epoch


def main():
    last_epoch = get_last_epoch()
    remaining = TOTAL_EPOCHS - (last_epoch + 1)

    if remaining <= 0:
        print("🎉 Tous les epochs ont déjà été exécutés !")
        return

    num_blocks = math.ceil(remaining / BLOCK_SIZE)

    print(f"\n🚀 Objectif : {TOTAL_EPOCHS} epochs")
    print(f"🔁 Reprise à epoch {last_epoch+1}")
    print(f"📦 Épochs restants : {remaining}")
    print(f"🧮 Blocs nécessaires : {num_blocks}\n")

    for b in range(num_blocks):
        print("=" * 60)
        print(f"🔄 Lancement du bloc {b+1}/{num_blocks}")

        subprocess.run([
            "python", "train_block.py",
            "--data", DATA,
            "--name", RUN_NAME,
            "--base-model", BASE_MODEL,
            "--epochs-block", str(BLOCK_SIZE),
        ], check=True)

        last_epoch = get_last_epoch()
        print(f"📈 Nouveau last_epoch = {last_epoch}")

    print("\n" + "=" * 60)
    print(f"🎉 Entraînement complet terminé ({TOTAL_EPOCHS} epochs).")
    print(f"🏁 Modèle final : runs/detect/{RUN_NAME}/weights/best.pt")


if __name__ == "__main__":
    main()
