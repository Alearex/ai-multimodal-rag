import os
import requests
from dotenv import load_dotenv

# Charge les variables d'environnement du fichier .env
load_dotenv()

TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY")
if not TOGETHER_API_KEY:
    raise RuntimeError(
        "Missing TOGETHER_API_KEY. Create a .env file with 'TOGETHER_API_KEY=your-key' and reload."
    )

def query_mixtral(prompt, temperature=0.7, max_tokens=512):
    """
    Interroge le modèle Mixtral via l'API Together.

    Args:
        prompt (str): question + contexte
        temperature (float): contrôle la créativité
        max_tokens (int): nombre max de tokens générés

    Returns:
        str: texte généré par Mixtral
    """
    url = "https://api.together.xyz/v1/completions"
    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "mistralai/Mistral-7B-Instruct-v0.1",
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": 0.9,
        "stop": None
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()["choices"][0]["text"].strip()
    else:
        raise Exception(f"[{response.status_code}] {response.text}")