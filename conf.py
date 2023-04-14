import os

CODE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(CODE_DIR, 'data')

COLLECTION_NAME = "papers"
QUERY = "all"
LIMIT = 1000
BATCH_SIZE = 64

QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = os.environ.get("QDRANT_PORT", 6333)
