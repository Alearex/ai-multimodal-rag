# Pipeline RAG multimodal avec gestion de deux index FAISS séparés (texte et image)

import os
import tempfile
import numpy as np

from loaders.pdf_loader import load_pdf
from loaders.image_loader import load_image
from loaders.audio_loader import load_audio

from embeddings.embed_text import embed_text_chunks
from embeddings.embed_image import embed_image
from embeddings.embed_audio import embed_audio_text

from vector_store.faiss_manager import FaissManager
from utils.mixtral_api import query_mixtral

# Limite maximale de caractères pour chaque document inséré dans le prompt
MAX_DOC_LENGTH = 1000

def crop_text(text, max_len=MAX_DOC_LENGTH):
    """Tronque le texte s'il dépasse ``max_len`` et ajoute une indication."""
    if len(text) > max_len:
        return text[:max_len] + " ...[cropped]"
    return text

# Initialise deux index FAISS séparés : un pour le texte, un pour les images
text_index = FaissManager("vector_store/index_text", dim=384)   # MiniLM
image_index = FaissManager("vector_store/index_image", dim=512)  # CLIP

def reset_indexes():
    """Supprime toutes les données des deux index."""
    text_index.clear()
    image_index.clear()
    
def normalize(scores):
    """Normalise les distances en scores (1 = plus proche)."""
    max_val = max(scores) if scores else 1.0
    return [1 - (s / max_val) for s in scores]

def combined_search(text_index, image_index, text_query_vec, image_query_vec, k=5):
    """
    Effectue une recherche multimodale (texte + image) et fusionne les résultats.

    Args:
        text_index (FaissManager): index texte (MiniLM)
        image_index (FaissManager): index image (CLIP)
        text_query_vec (np.ndarray): vecteur de requête texte de forme ``(384,)`` ou ``(n, 384)``
        image_query_vec (np.ndarray): vecteur de requête image CLIP de forme ``(512,)`` ou ``(n, 512)``

        k (int): nombre de résultats à retourner par index

    Returns:
        list[dict]: liste fusionnée des résultats triés par score décroissant
    """
    text_results = text_index.search(text_query_vec, k)
    image_results = image_index.search(image_query_vec, k)

    t_scores = normalize([score for _, score in text_results])
    i_scores = normalize([score for _, score in image_results])

    combined = [
        {"source": "text", "content": doc, "score": t_scores[i]}
        for i, (doc, _) in enumerate(text_results)
    ] + [
        {"source": "image", "content": img, "score": i_scores[i]}
        for i, (img, _) in enumerate(image_results)
    ]

    combined.sort(key=lambda x: x["score"], reverse=True)
    return combined

def process_query(uploaded_files, question):
    """
    Traite les fichiers uploadés, extrait les contenus (texte, image, audio),
    les encode et les ajoute dans les index FAISS, puis effectue une recherche RAG multimodale.

    Args:
        uploaded_files (list): fichiers uploadés par l'utilisateur (PDF, image, audio)
        question (str): question posée par l'utilisateur

    Returns:
        str: réponse générée par Mixtral sur la base des documents retrouvés
    """
    for file in uploaded_files:
        suffix = file.name.split(".")[-1].lower()

        with tempfile.NamedTemporaryFile(delete=False, suffix="." + suffix) as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        # Traitement PDF : texte par page
        if suffix == "pdf":
            texts = [crop_text(t) for t in load_pdf(tmp_path)]
            vectors = embed_text_chunks(texts)
            text_index.add_embeddings(texts, vectors)

        # Traitement image : caption + embedding CLIP
        elif suffix in ["png", "jpg", "jpeg"]:
            caption = load_image(tmp_path)
            print("Caption généré pour l'image :", caption)
            vector = embed_image(caption)
            image_index.add_embeddings([caption], [vector])

        # Traitement audio : transcription + embedding MiniLM
        elif suffix in ["mp3", "wav"]:
            transcript = crop_text(load_audio(tmp_path))
            vector = embed_audio_text(transcript)
            text_index.add_embeddings([transcript], [vector])

        os.unlink(tmp_path)  # Nettoyage du fichier temporaire

    # --- Encodage de la question ---
    # Texte : encode avec MiniLM → tensor(1, 384) → .numpy() pour FAISS
    text_query_vec = embed_text_chunks([question])[0].detach().cpu().numpy()

    # Image : encode avec CLIP (retourne directement un vecteur np.ndarray shape (512,))
    # La recherche accepte maintenant ce format 1D directement
    image_query_vec = embed_image(question)

    # Recherche fusionnée texte + image
    results = combined_search(text_index, image_index, text_query_vec, image_query_vec, k=5)

    # Construction du contexte en séparant clairement chaque document
    formatted = []
    for idx, r in enumerate(results, start=1):
        snippet = crop_text(r["content"])
        formatted.append(f"### Document {idx} ({r['source']})\n{snippet}")

    context = "\n\n".join(formatted)
    prompt = f"{question}\n\n{context}"
    return query_mixtral(prompt)