from .embed_text import embed_text_chunks

def embed_audio_text(text):
    """
    Encode une transcription audio en vecteur d'embedding texte.

    Args:
        text (str): transcription du contenu audio

    Returns:
        torch.Tensor | np.ndarray: vecteur unique représentant le texte
    """
    # Encodage comme un seul chunk (liste de taille 1)
    return embed_text_chunks([text])[0]
