from faster_whisper import WhisperModel

# Chargement du modèle Whisper (base, rapide sur CPU/GPU)
model = WhisperModel("base", compute_type="auto")  # auto = fp16 sur GPU, int8/cpu sinon

def load_audio(audio_path):
    """
    Transcrit un fichier audio en texte avec Whisper (rapide).
    
    Args:
        audio_path (str): chemin vers un fichier audio (.mp3, .wav, etc.)
    
    Returns:
        str: transcription complète du contenu audio
    """
    segments, _ = model.transcribe(audio_path, beam_size=5)  # beam_size améliore la qualité

    return " ".join(seg.text.strip() for seg in segments)
