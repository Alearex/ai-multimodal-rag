# Importation de Streamlit, le framework pour créer des interfaces web simples en Python
import streamlit as st

# On importe notre fonction principale de traitement, définie dans rag_pipeline.py
from rag_pipeline import process_query, reset_indexes

# Configuration de la page web (nom de l'onglet du navigateur, icône, etc.)
st.set_page_config(page_title="Assistant IA Multimodal RAG")

# Titre affiché en haut de la page
st.title("📁 Assistant IA Multimodal RAG")

# Permet à l'utilisateur d'envoyer un ou plusieurs fichiers (PDF, image, audio)
uploaded_files = st.file_uploader(
    label="Upload files (PDF, image, audio)",                # Message à l'utilisateur
    type=["pdf", "png", "jpg", "jpeg", "mp3", "wav"],        # Types autorisés
    accept_multiple_files=True                               # Permet plusieurs fichiers
)

# Champ texte pour poser une question à l'IA
question = st.text_input("Posez votre question :")

# Bouton pour réinitialiser les index
if st.button("Réinitialiser les index"):
    reset_indexes()
    st.success("Index réinitialisés")
    
# Bouton "Lancer la recherche" : une fois cliqué, on traite les fichiers + la question
if st.button("Lancer la recherche") and uploaded_files and question:
    # Affiche un message de chargement pendant le traitement
    with st.spinner("Traitement en cours..."):
        # Appelle la fonction principale du backend pour analyser les fichiers + question
        answer = process_query(uploaded_files, question)
    
    # Affiche le résultat sous forme de texte lisible
    st.markdown("### Réponse :")
    st.write(answer)
