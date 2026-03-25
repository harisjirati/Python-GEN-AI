import chromadb
from chromadb.config import Settings
import numpy as np
import uuid


class VectorStore:
    def __init__(self, dim: int, persist_directory: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name="pdf_chunks",
            metadata={"hnsw:space": "cosine"}
        )

    def add(self, embeddings, chunks: list, metadata: list = None):
        if len(embeddings) == 0:
            return

        ids = [str(uuid.uuid4()) for _ in chunks]

        if metadata is None:
            metadata = [{"source": "unknown", "chunk_index": i} for i in range(len(chunks))]

        self.collection.add(
            ids=ids,
            embeddings=[emb.tolist() if hasattr(emb, 'tolist') else emb for emb in embeddings],
            documents=chunks,
            metadatas=metadata
        )

    def search(self, query_embedding, k: int = 5) -> list:
        """Return top-k text chunks."""
        query_vec = query_embedding.tolist() if hasattr(query_embedding, 'tolist') else query_embedding

        results = self.collection.query(
            query_embeddings=[query_vec],
            n_results=min(k, self.collection.count())
        )

        return results["documents"][0] if results["documents"] else []

    def get_sources(self, query_embedding, k: int = 5) -> list:
        """
        Return unique source filenames from the top-k results.
        Used to show citations alongside the answer in the UI.
        """
        query_vec = query_embedding.tolist() if hasattr(query_embedding, 'tolist') else query_embedding

        results = self.collection.query(
            query_embeddings=[query_vec],
            n_results=min(k, self.collection.count()),
            include=["metadatas"]
        )

        if not results["metadatas"]:
            return []

        # Deduplicate — multiple chunks may come from the same file
        seen = set()
        unique_sources = []
        for meta in results["metadatas"][0]:
            src = meta.get("source", "unknown")
            if src not in seen:
                seen.add(src)
                unique_sources.append(src)

        return unique_sources

    def count(self) -> int:
        return self.collection.count()

    def clear(self):
        self.client.delete_collection("pdf_chunks")
        self.collection = self.client.get_or_create_collection(
            name="pdf_chunks",
            metadata={"hnsw:space": "cosine"}
        )

    def is_populated(self) -> bool:
        return self.collection.count() > 0