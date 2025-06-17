from transformers import AutoProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

# Chargement du processor et du modèle BLIP (base)
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
model.eval()  # optionnel mais explicite : passe le modèle en mode évaluation

def load_image(image_path):
    """
    Génère une légende pour une image avec le modèle BLIP.
    
    Args:
        image_path (str): chemin vers l'image
    
    Returns:
        str: légende générée
    """
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        output = model.generate(**inputs)

    return processor.batch_decode(output, skip_special_tokens=True)[0]
