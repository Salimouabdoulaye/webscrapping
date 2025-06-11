#!/usr/bin/env python3
"""
Script de lancement complet pour l'analyseur de sentiments Amazon France
Ce script vérifie l'environnement et lance l'application Streamlit
"""

import os
import sys
import subprocess
import importlib.util
from pathlib import Path

def check_python_version():
    """Vérifie la version de Python"""
    if sys.version_info < (3, 8):
        print("ERREUR: Python 3.8+ requis. Version actuelle:", sys.version)
        return False
    print(f"OK: Python {sys.version_info.major}.{sys.version_info.minor} détecté")
    return True

def check_virtual_env():
    """Vérifie si on est dans un environnement virtuel"""
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    if in_venv:
        print("OK: Environnement virtuel activé")
    else:
        print("ATTENTION: Pas d'environnement virtuel détecté (recommandé mais pas obligatoire)")
    return True

def check_required_files():
    """Vérifie que tous les fichiers requis sont présents"""
    required_files = [
        'app.py',
        'requirements.txt',
        'modele.py'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
        else:
            print(f"OK: {file} trouvé")
    
    if missing_files:
        print(f"ERREUR: Fichiers manquants: {', '.join(missing_files)}")
        return False
    
    return True

def check_model_files():
    """Vérifie la présence des fichiers de modèle"""
    pkl_files = list(Path('.').glob('pipeline_amazon_fr_*.pkl'))
    json_files = list(Path('.').glob('metadata_amazon_fr_*.json'))
    
    if not pkl_files:
        print("ERREUR: Aucun fichier de modèle (.pkl) trouvé!")
        print("   Exécutez d'abord votre script modele.py pour entraîner le modèle")
        return False
    
    print(f"OK: Modèle trouvé: {pkl_files[0].name}")
    
    if json_files:
        print(f"OK: Métadonnées trouvées: {json_files[0].name}")
    else:
        print("ATTENTION: Fichier de métadonnées non trouvé (optionnel)")
    
    return True

def install_requirements():
    """Installe les dépendances si nécessaire"""
    try:
        # Vérifier si streamlit est installé
        import streamlit
        print("OK: Streamlit déjà installé")
        return True
    except ImportError:
        print("INSTALLATION: Installation des dépendances...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
            print("OK: Dépendances installées avec succès")
            return True
        except subprocess.CalledProcessError as e:
            print(f"ERREUR: Erreur lors de l'installation: {e}")
            return False

def check_streamlit_installation():
    """Vérifie que Streamlit est correctement installé"""
    try:
        import streamlit as st
        print(f"OK: Streamlit version {st.__version__} installé")
        return True
    except ImportError:
        print("ERREUR: Streamlit n'est pas installé")
        return False

def launch_streamlit():
    """Lance l'application Streamlit"""
    print("\nLANCEMENT: Démarrage de l'application Streamlit...")
    print("NAVIGATEUR: L'application s'ouvrira dans votre navigateur")
    print("URL: http://localhost:8501")
    print("ARRÊT: Pour arrêter, appuyez sur Ctrl+C dans ce terminal\n")
    
    try:
        subprocess.run([sys.executable, '-m', 'streamlit', 'run', 'app.py'])
    except KeyboardInterrupt:
        print("\nARRÊT: Application arrêtée par l'utilisateur")
    except Exception as e:
        print(f"\nERREUR: Erreur lors du lancement: {e}")

def main():
    """Fonction principale"""
    print("ANALYSEUR DE SENTIMENTS AMAZON FRANCE - LANCEMENT")
    print("=" * 60)
    
    # Vérifications préliminaires
    checks = [
        ("Version Python", check_python_version),
        ("Environnement virtuel", check_virtual_env),
        ("Fichiers requis", check_required_files),
        ("Fichiers de modèle", check_model_files),
    ]
    
    print("\nVÉRIFICATIONS PRÉLIMINAIRES")
    print("-" * 40)
    
    all_checks_passed = True
    for check_name, check_func in checks:
        print(f"\n[{check_name}]")
        if not check_func():
            all_checks_passed = False
    
    if not all_checks_passed:
        print("\nERREUR: Certaines vérifications ont échoué. Corrigez les erreurs avant de continuer.")
        return
    
    print("\nINSTALLATION DES DÉPENDANCES")
    print("-" * 40)
    
    if not install_requirements():
        print("ERREUR: Échec de l'installation des dépendances")
        return
    
    if not check_streamlit_installation():
        print("ERREUR: Streamlit n'est pas disponible après installation")
        return
    
    print("\nSUCCÈS: TOUTES LES VÉRIFICATIONS SONT PASSÉES")
    print("=" * 60)
    
    # Demander confirmation
    response = input("\nLancer l'application Streamlit ? (y/N): ").lower().strip()
    
    if response in ['y', 'yes', 'oui', 'o']:
        launch_streamlit()
    else:
        print("ANNULÉ: Lancement annulé. Pour lancer manuellement:")
        print("   streamlit run app.py")

if __name__ == "__main__":
    main()