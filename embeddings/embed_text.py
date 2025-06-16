from sentence_transformers import SentenceTransformer

# Chargement d'un modèle léger et rapide (toujours pertinent)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed_text_chunks(chunks):
    """
    Encode une liste de textes (phrases, paragraphes...) en embeddings.

    Args:
        chunks (list[str]): liste de textes (chaque texte = un chunk)

    Returns:
        torch.Tensor: tenseur d'embeddings (un vecteur par chunk)
    """
    return model.encode(chunks, convert_to_tensor=True, normalize_embeddings=True)
