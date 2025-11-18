# CADOTProject

Description du projet
Ce projet vise à améliorer la détection d’objets sur des images aériennes de Paris issues du défi cadot.onrender.com. Pour cela, nous fine-tunons un modèle YOLO sur l’ensemble des images annotées afin d’obtenir une détection robuste des catégories d’objets présentes en milieu urbain dense.
Après entraînement, nous analysons précisément les performances du modèle pour identifier les classes les moins bien détectées (faible précision, rappel ou mAP).
Pour remédier au manque de représentativité de ces objets, nous générons de nouvelles données synthétiques grâce à des modèles de diffusion (Stable Diffusion) spécialisés via des LoRA, permettant de produire des variantes réalistes d’objets rares ou difficiles.
Ces données augmentées sont ensuite réinjectées dans un second fine-tuning de YOLO, avec pour objectif de renforcer la performance globale et d’améliorer significativement la détection des classes initialement défaillantes.