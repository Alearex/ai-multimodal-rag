import faiss
import numpy as np
import os
import pickle

class FaissManager:
    """
    Gestionnaire FAISS pour indexer, rechercher et sauvegarder des embeddings.
    """

    def __init__(self, index_path, dim=384):
        """
        Initialise ou charge un index FAISS.

        Args:
            index_path (str): dossier de sauvegarde de l’index
            dim (int): dimension des vecteurs (par défaut : 384 pour MiniLM)
        """
        self.index_path = index_path
        self.index_file = os.path.join(index_path, "faiss.index")
        self.store_file = os.path.join(index_path, "store.pkl")
        self.dim = dim

        os.makedirs(index_path, exist_ok=True)

        if os.path.exists(self.index_file):
            self.index = faiss.read_index(self.index_file)
            with open(self.store_file, "rb") as f:
                self.texts = pickle.load(f)
        else:
            self.index = faiss.IndexFlatL2(dim)  # Index L2 (cosine possible si normalisé)
            self.texts = []

    def add_embeddings(self, texts, embeddings):
        """
        Ajoute des vecteurs et leurs textes à l’index.

        Args:
            texts (list[str]): textes liés aux vecteurs
            embeddings (list[np.ndarray] ou torch.Tensor): vecteurs
        """
        embeddings_np = np.stack([
            e.detach().cpu().numpy() if hasattr(e, 'detach') else np.asarray(e)
            for e in embeddings
        ])

        self.index.add(embeddings_np)
        self.texts.extend(texts)
        self.save()

    def save(self):
        """Sauvegarde l’index et les textes associés."""
        faiss.write_index(self.index, self.index_file)
        with open(self.store_file, "wb") as f:
            pickle.dump(self.texts, f)

    def search(self, query_embedding, k=5):
        """
        Recherche les k textes les plus proches d’un vecteur.

        Args:
            query_embedding (np.ndarray ou torch.Tensor): vecteur de requête
            k (int): nombre de résultats à retourner

        Returns:
            list[str]: textes les plus proches
        """
        query_np = np.asarray(query_embedding.detach().cpu().numpy() if hasattr(query_embedding, 'detach') else query_embedding)
        query_np = np.expand_dims(query_np, axis=0)  # forme (1, dim)

        _, indices = self.index.search(query_np, k)

        return [self.texts[i] for i in indices[0] if i < len(self.texts)]
