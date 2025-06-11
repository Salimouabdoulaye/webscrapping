import streamlit as st
import joblib
import pandas as pd
import numpy as np
import json
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import os
import io
import sys

# Imports pour le preprocessing (nécessaires pour charger le modèle)
from sklearn.base import BaseEstimator, TransformerMixin
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import string
import unicodedata
import re

# Téléchargement des ressources NLTK si nécessaire
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

# CLASSE NÉCESSAIRE POUR LE CHARGEMENT DU MODÈLE
class FrenchTextPreprocessor(BaseEstimator, TransformerMixin):
    """
    Preprocessor spécialisé pour le texte français
    Cette classe DOIT être identique à celle utilisée lors de l'entraînement
    """
    def __init__(self, remove_stopwords=True, stem=True, min_length=2):
        self.remove_stopwords = remove_stopwords
        self.stem = stem
        self.min_length = min_length
        
        # Initialisation des outils
        try:
            self.french_stopwords = set(stopwords.words('french'))
        except:
            self.french_stopwords = set()
        
        self.stemmer = SnowballStemmer('french')
        
        # Stopwords supplémentaires spécifiques à Amazon
        amazon_stopwords = {
            'lire', 'plus', 'produit', 'amazon', 'achat', 'acheter', 'commander',
            'livraison', 'prix', 'euro', 'euros', 'article', 'site', 'web',
            'internet', 'online', 'boutique', 'magasin', 'vendeur', 'client'
        }
        self.french_stopwords.update(amazon_stopwords)
        
    def remove_accents(self, text):
        """Supprime les accents du texte"""
        return ''.join(c for c in unicodedata.normalize('NFD', text)
                      if unicodedata.category(c) != 'Mn')
    
    def clean_text(self, text):
        """Nettoie le texte"""
        if pd.isna(text):
            return ""
        
        # Conversion en string et minuscules
        text = str(text).lower()
        
        # Suppression des URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Suppression des mentions email
        text = re.sub(r'\S+@\S+', '', text)
        
        # Suppression des caractères spéciaux mais garde les espaces
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Suppression des chiffres isolés
        text = re.sub(r'\b\d+\b', '', text)
        
        # Suppression des espaces multiples
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def preprocess_text(self, text):
        """Pipeline de preprocessing complet"""
        # Nettoyage initial
        text = self.clean_text(text)
        
        if not text:
            return ""
        
        try:
            # Tokenisation
            tokens = word_tokenize(text, language='french')
            
            # Filtrage des tokens
            processed_tokens = []
            for token in tokens:
                # Filtrer les tokens trop courts
                if len(token) < self.min_length:
                    continue
                    
                # Suppression des stopwords
                if self.remove_stopwords and token in self.french_stopwords:
                    continue
                    
                # Stemming
                if self.stem:
                    token = self.stemmer.stem(token)
                
                processed_tokens.append(token)
            
            return ' '.join(processed_tokens)
            
        except Exception as e:
            print(f"Erreur lors du preprocessing: {e}")
            return text
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """Transforme une série de textes"""
        if isinstance(X, pd.Series):
            return X.apply(self.preprocess_text)
        elif isinstance(X, list):
            return [self.preprocess_text(text) for text in X]
        else:
            return [self.preprocess_text(str(text)) for text in X]

# Configuration de la page
st.set_page_config(
    page_title="Analyseur de Sentiments Amazon FR",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF6B35;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 8px solid;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .positive {
        background-color: #d4edda;
        border-left-color: #28a745;
        color: #155724;
    }
    .negatif {
        background-color: #f8d7da;
        border-left-color: #dc3545;
        color: #721c24;
    }
    .neutre {
        background-color: #fff3cd;
        border-left-color: #ffc107;
        color: #856404;
    }
    .stButton > button {
        background-color: #FF6B35;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .metric-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border: 1px solid #dee2e6;
    }
    .loading-box {
        background-color: #e7f3ff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #007bff;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def find_model_files():
    """
    Trouve les fichiers de modèle sans les charger
    """
    pkl_files = list(Path('.').glob('pipeline_amazon_fr_*.pkl'))
    json_files = list(Path('.').glob('metadata_amazon_fr_*.json'))
    
    if not pkl_files:
        return None, None, None
    
    # Prendre le plus récent
    latest_model = max(pkl_files, key=os.path.getctime)
    
    # Chercher le fichier de métadonnées correspondant
    timestamp = str(latest_model).split('_')[-1].replace('.pkl', '')
    metadata_file = f'metadata_amazon_fr_{timestamp}.json'
    
    return str(latest_model), metadata_file, timestamp

@st.cache_resource
def load_model_safe(model_path):
    """
    Charge le modèle de manière sécurisée avec gestion d'erreur
    """
    try:
        with st.spinner(f"Chargement du modèle: {os.path.basename(model_path)}..."):
            pipeline = joblib.load(model_path)
        return pipeline, None
    except Exception as e:
        error_msg = f"Erreur lors du chargement du modèle: {str(e)}"
        return None, error_msg

def load_metadata_safe(metadata_path):
    """
    Charge les métadonnées de manière sécurisée
    """
    try:
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f), None
        else:
            return {}, "Fichier de métadonnées non trouvé"
    except Exception as e:
        return {}, f"Erreur lors du chargement des métadonnées: {str(e)}"

def predict_sentiment(pipeline, text):
    """
    Prédit le sentiment d'un texte avec gestion d'erreur améliorée
    """
    try:
        if not text or not text.strip():
            return None, None, "Texte vide"
        
        # Prédiction
        prediction = pipeline.predict([text])[0]
        
        # Essayer d'obtenir les probabilités
        probabilities = None
        try:
            proba = pipeline.predict_proba([text])[0]
            classes = pipeline.classes_
            probabilities = dict(zip(classes, proba))
        except AttributeError:
            # Le modèle n'a pas de predict_proba
            pass
        except Exception as e:
            st.warning(f"Impossible d'obtenir les probabilités: {e}")
        
        return prediction, probabilities, None
    
    except Exception as e:
        return None, None, f"Erreur lors de la prédiction: {str(e)}"

def display_prediction(prediction, probabilities=None):
    """
    Affiche la prédiction avec style
    """
    if prediction is None:
        return
    
    # Configuration des styles par catégorie
    config = {
        'positive': {
            'symbol': '[+]',
            'color': '#28a745',
            'label': 'Positif',
            'description': 'Ce commentaire exprime une satisfaction'
        },
        'negatif': {
            'symbol': '[-]',
            'color': '#dc3545',
            'label': 'Négatif',
            'description': 'Ce commentaire exprime une insatisfaction'
        },
        'neutre': {
            'symbol': '[=]',
            'color': '#ffc107',
            'label': 'Neutre',
            'description': 'Ce commentaire est neutre ou mitigé'
        }
    }
    
    if prediction in config:
        conf = config[prediction]
        
        st.markdown(f"""
        <div class="prediction-box {prediction}">
            <h2 style="margin: 0; font-size: 1.8rem;">
                {conf['symbol']} Sentiment: {conf['label']}
            </h2>
            <p style="margin: 0.5rem 0 0 0; font-style: italic;">
                {conf['description']}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Afficher les probabilités si disponibles
        if probabilities:
            st.subheader("Confiance du modèle")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Graphique en barres
                df_prob = pd.DataFrame(list(probabilities.items()), 
                                     columns=['Catégorie', 'Probabilité'])
                df_prob['Probabilité'] = df_prob['Probabilité'] * 100
                
                fig = px.bar(df_prob, x='Catégorie', y='Probabilité', 
                           title="Distribution des probabilités (%)",
                           color='Probabilité',
                           color_continuous_scale='RdYlGn',
                           text='Probabilité')
                
                fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                fig.update_layout(showlegend=False, height=400)
                fig.update_yaxis(range=[0, 100])
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Métriques détaillées
                st.write("**Probabilités détaillées:**")
                for cat, prob in probabilities.items():
                    st.metric(cat.capitalize(), f"{prob*100:.1f}%")

def analyze_batch_safe(pipeline, df):
    """
    Analyse un DataFrame de commentaires avec barre de progression
    """
    if 'review_text' not in df.columns:
        st.error("La colonne 'review_text' est requise dans votre fichier CSV")
        return None
    
    # Créer une barre de progression
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    predictions = []
    errors = 0
    
    total = len(df)
    
    for i, text in enumerate(df['review_text']):
        # Mettre à jour la progression
        progress = (i + 1) / total
        progress_bar.progress(progress)
        status_text.text(f'Analyse en cours... {i+1}/{total} commentaires traités')
        
        pred, _, error = predict_sentiment(pipeline, str(text))
        if error:
            errors += 1
            predictions.append('erreur')
        else:
            predictions.append(pred if pred else 'inconnu')
    
    # Nettoyer la barre de progression
    progress_bar.empty()
    status_text.empty()
    
    if errors > 0:
        st.warning(f"Attention: {errors} erreurs détectées lors de l'analyse")
    
    df['sentiment_predit'] = predictions
    return df

def main():
    """
    Application principale avec gestion d'erreur améliorée
    """
    
    # Header
    st.markdown('<h1 class="main-header">Analyseur de commentaire Amazon</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem; color: #666;">
        Analysez le sentiment de vos commentaires clients Amazon avec notre IA spécialisée
    </div>
    """, unsafe_allow_html=True)
    
    # Étape 1: Trouver les fichiers
    model_path, metadata_path, timestamp = find_model_files()
    
    if not model_path:
        st.error("Aucun fichier de modèle trouvé!")
        st.markdown("""
        <div class="loading-box">
            <strong>Pour résoudre ce problème:</strong><br>
            1. Assurez-vous d'avoir exécuté votre script 'modele.py'<br>
            2. Vérifiez la présence du fichier 'pipeline_amazon_fr_*.pkl'<br>
            3. Redémarrez l'application Streamlit
        </div>
        """, unsafe_allow_html=True)
        st.stop()
    
    # Étape 2: Afficher les informations sur les fichiers trouvés
    st.info(f"Modèle détecté: {os.path.basename(model_path)}")
    
    # Étape 3: Charger le modèle
    if 'pipeline' not in st.session_state:
        pipeline, error = load_model_safe(model_path)
        
        if error:
            st.error(error)
            st.markdown("""
            <div class="loading-box">
                <strong>Solutions possibles:</strong><br>
                1. Vérifiez que le fichier n'est pas corrompu<br>
                2. Réentraînez le modèle avec le script 'modele.py'<br>
                3. Vérifiez la compatibilité des versions de scikit-learn
            </div>
            """, unsafe_allow_html=True)
            st.stop()
        
        st.session_state.pipeline = pipeline
        st.success("Modèle chargé avec succès!")
    
    pipeline = st.session_state.pipeline
    
    # Étape 4: Charger les métadonnées
    metadata, meta_error = load_metadata_safe(metadata_path)
    if meta_error:
        st.warning(meta_error)
    
    # Sidebar avec informations du modèle
    with st.sidebar:
        st.header("Informations du Modèle")
        
        # Informations de base
        st.markdown(f"""
        <div class="metric-container">
            <strong>Fichier:</strong> {os.path.basename(model_path)}<br>
            <strong>Timestamp:</strong> {timestamp}<br>
            <strong>Taille:</strong> {os.path.getsize(model_path) / 1024 / 1024:.1f} MB
        </div>
        """, unsafe_allow_html=True)
        
        if metadata:
            st.markdown(f"""
            <div class="metric-container">
                <strong>Modèle:</strong> {metadata.get('model_name', 'N/A')}<br>
                <strong>F1 Score:</strong> {metadata.get('f1_score', 0):.3f}<br>
                <strong>Accuracy:</strong> {metadata.get('accuracy', 0):.3f}<br>
                <strong>Données d'entraînement:</strong> {metadata.get('train_size', 'N/A')}<br>
                <strong>Features:</strong> {metadata.get('features_count', 'N/A')}
            </div>
            """, unsafe_allow_html=True)
            
            if 'classes' in metadata:
                st.write("**Classes détectées:**")
                for classe in metadata['classes']:
                    st.write(f"• {classe.capitalize()}")
        
        st.header("Guide d'utilisation")
        st.markdown("""
        **Analyse Simple:**
        - Saisissez un commentaire dans la zone de texte
        - Cliquez sur "Analyser le sentiment"
        
        **Analyse de Masse:**
        - Uploadez un fichier CSV avec une colonne 'review_text'
        - Téléchargez les résultats
        
        **Exemples:**
        - Testez avec des exemples prédéfinis
        """)
    
    # Onglets principaux
    tab1, tab2, tab3 = st.tabs(["Analyse Simple", "Analyse de Masse", "Exemples"])
    
    with tab1:
        st.header("Analysez un commentaire")
        
        # Zone de texte
        user_input = st.text_area(
            "Saisissez votre commentaire client:",
            placeholder="Exemple: Ce produit est fantastique, je le recommande vivement!",
            height=150
        )
        
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("Analyser le sentiment", type="primary"):
                if user_input.strip():
                    prediction, probabilities, error = predict_sentiment(pipeline, user_input)
                    if error:
                        st.error(error)
                    elif prediction:
                        display_prediction(prediction, probabilities)
                else:
                    st.warning("Veuillez saisir un commentaire à analyser.")
        
        with col2:
            if st.button("Effacer"):
                st.rerun()
    
    with tab2:
        st.header("Analyse de masse")
        st.write("Uploadez un fichier CSV contenant une colonne 'review_text' pour analyser plusieurs commentaires.")
        
        uploaded_file = st.file_uploader(
            "Choisissez votre fichier CSV",
            type=['csv'],
            help="Le fichier doit contenir une colonne 'review_text'"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.write(f"Fichier chargé: {len(df)} lignes")
                
                # Aperçu des données
                st.subheader("Aperçu des données")
                st.dataframe(df.head())
                
                if st.button("Analyser tous les commentaires"):
                    df_analyzed = analyze_batch_safe(pipeline, df)
                    
                    if df_analyzed is not None:
                        st.success("Analyse terminée!")
                        
                        # Afficher les résultats
                        st.subheader("Résultats de l'analyse")
                        
                        # Statistiques
                        sentiment_counts = df_analyzed['sentiment_predit'].value_counts()
                        
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            # Graphique en secteurs
                            fig_pie = px.pie(values=sentiment_counts.values, 
                                           names=sentiment_counts.index,
                                           title="Distribution des sentiments")
                            st.plotly_chart(fig_pie, use_container_width=True)
                        
                        with col2:
                            # Métriques
                            for sentiment, count in sentiment_counts.items():
                                percentage = (count / len(df_analyzed)) * 100
                                st.metric(f"{sentiment.capitalize()}", 
                                        f"{count} ({percentage:.1f}%)")
                        
                        # Tableau des résultats
                        st.subheader("Tableau des résultats")
                        st.dataframe(df_analyzed)
                        
                        # Téléchargement
                        csv = df_analyzed.to_csv(index=False)
                        st.download_button(
                            label="Télécharger les résultats (CSV)",
                            data=csv,
                            file_name=f"resultats_analyse_sentiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                        
            except Exception as e:
                st.error(f"Erreur lors du chargement du fichier: {e}")
    
    with tab3:
        st.header("Exemples de test")
        st.write("Testez le modèle avec ces exemples prédéfinis:")
        
        exemples = [
            ("Commentaire très positif", "Ce produit est absolument fantastique! Qualité exceptionnelle, livraison rapide, je recommande vivement. Parfait!"),
            ("Commentaire négatif", "Très déçu de cet achat. La qualité n'est pas au rendez-vous, le produit est arrivé cassé et le service client ne répond pas."),
        ]
        
        for titre, texte in exemples:
            with st.expander(f"[TEST] {titre}"):
                st.write(f"**Texte:** {texte}")
                
                if st.button(f"Analyser", key=f"btn_{titre}"):
                    prediction, probabilities, error = predict_sentiment(pipeline, texte)
                    if error:
                        st.error(error)
                    elif prediction:
                        display_prediction(prediction, probabilities)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        Analyseur de commentaires Amazon - Propulsé par Machine Learning
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
