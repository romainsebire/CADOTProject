
#!/bin/zsh

# Script d'activation de l'environnement virtuel
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "Environnement .venv activé"
else
    echo "ERREUR: Le dossier .venv n'existe pas"
    echo "Créez d'abord l'environnement avec: python3 -m venv .venv"
    exit 1
fi
