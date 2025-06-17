import torch
import numpy as np
from open_clip import create_model_and_transforms, tokenize

# Chargement du modèle CLIP + préprocesseurs associés
model, _, preprocess = create_model_and_transforms(
    model_name="ViT-B-32",
    pretrained="openai"
)

# Passage explicite en mode évaluation
model.eval()

# Passage sur GPU si dispo (optionnel mais utile)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def embed_image(text):
    """
    Encode une description d'image en embedding CLIP (texte).

    Args:
        text (str): description textuelle associée à l'image

    Returns:
        np.ndarray: vecteur CLIP 512D normalisé
    """
    # Tokenisation du texte et transfert sur le bon device
    tokens = tokenize([text]).to(device)

    with torch.no_grad():
        embedding = model.encode_text(tokens)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)  # Normalisation L2

    return embedding.squeeze(0).cpu().numpy()
