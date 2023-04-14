from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from conf import QDRANT_HOST, QDRANT_PORT
from typing import List


class NeuralSearch:
    """
    Handler for neural search operations
    """
    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        self.model = SentenceTransformer("distilbert-base-nli-stsb-mean-tokens", device="cpu")
        self.qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    
    def search(self, text: str) -> List[dict]:
        """Function to search and filter in collection
        """
        vector = self.model.encode(text).tolist()
        hits = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=vector,
            limit=10
        )

        return [hit.payload for hit in hits]
