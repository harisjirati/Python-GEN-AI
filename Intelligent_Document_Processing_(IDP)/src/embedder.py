from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self):
        # Lightweight + fast model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def embed(self, texts):
        """
        Convert list of text chunks into embeddings
        """
        return self.model.encode(texts)