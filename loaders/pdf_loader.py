import fitz  # PyMuPDF

def load_pdf(path):
    """
    Ouvre un fichier PDF et retourne une liste de textes, un par page.
    """
    with fitz.open(path) as doc:
        return [page.get_text() for page in doc]
