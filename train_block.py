# train_block.py
import os
import argparse
import torch
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--base-model", type=str, required=True)
    parser.add_argument("--epochs-block", type=int, default=5)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=2)
    args = parser.parse_args()

    # Chemin du checkpoint existant
    checkpoint_path = os.path.join("runs", "detect", args.name, "weights", "last.pt")

    # Déterminer d'où reprendre
    if os.path.exists(checkpoint_path):
        print(f"📂 Reprise depuis : {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model_path = checkpoint_path
        last_epoch = ckpt.get("epoch", -1)
        print(f"🔁 Dernier epoch trouvé dans last.pt : {last_epoch}")
    else:
        print(f"📂 Premier bloc → modèle de base : {args.base_model}")
        model_path = args.base_model
        last_epoch = -1  # pas encore d'epoch effectuée

    # Calculer l’epoch de départ pour l’affichage
    global_start_epoch = last_epoch + 1
    global_end_epoch = global_start_epoch + args.epochs_block

    print(f"🚀 Bloc d'entraînement : epochs {global_start_epoch} → {global_end_epoch-1}")

    # Charger le modèle YOLO
    model = YOLO(model_path)

    # Entraînement du bloc
    results = model.train(
        data=args.data,
        epochs=args.epochs_block,
        imgsz=args.imgsz,
        batch=args.batch,
        device="mps",
        workers=1,
        name=args.name,
        cache=False,
        amp=False,
        save_period=-1,
        verbose=False,
        resume=False,
        plots=False,        # Désactive la génération de graphiques
        exist_ok=True,      # Évite les questions de confirmation
    )

    print(f"✅ Bloc terminé : modèle sauvegardé jusqu’à l’epoch {global_end_epoch-1}")

if __name__ == "__main__":
    main()
