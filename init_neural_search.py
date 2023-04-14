import arxiv
import json
import os
import numpy as np
import pandas as pd

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams
from sentence_transformers import SentenceTransformer
from conf import DATA_DIR, COLLECTION_NAME, QDRANT_HOST, QDRANT_PORT, LIMIT, BATCH_SIZE, QUERY


def prepare_data():
    """Function to retrieve data from arxiv and store in respective files
    """
    arxiv_search = arxiv.Search(
        query=QUERY,
        max_results=LIMIT,
        sort_by = arxiv.SortCriterion.SubmittedDate,
    )

    all_papers = []
    
    """
    result.entry_id: A url http://arxiv.org/abs/{id}.
    result.updated: When the result was last updated.
    result.published: When the result was originally published.
    result.title: The title of the result.
    result.summary: The result abstract.
    result.primary_category: The result's primary arXiv category. See arXiv: Category Taxonomy.
    """

    for result in arxiv_search.results():
        entry_dict = {
            "entry_id":result.entry_id,
            "updated":str(result.updated),
            "published":str(result.published),
            "title":result.title,
            "summary":result.summary,
            "primary_category":result.primary_category,
        }
        all_papers.append(entry_dict)

    all_papers_df = pd.DataFrame(all_papers)

    # create data folder 
    if not os.path.exists("data"): os.makedirs("data")

    # convert to json for payload 
    #json_df = all_papers_df.to_json(os.path.join(DATA_DIR, "papers.json"), orient="records")
    with open(os.path.join(DATA_DIR, "papers.json"), 'w', encoding='utf-8') as fd:
        json.dump(all_papers, fd, ensure_ascii=False, indent=4)

    # encode text to vectors and save vectors to file
    encoded_vecs = encode_to_vec(all_papers_df)
    np.save(os.path.join(DATA_DIR, 'summary_vectors.npy'), encoded_vecs, allow_pickle=False)


def encode_to_vec(data: pd.DataFrame) -> np.ndarray:
    """Function to encode arxiv paper titles and summaries to vectors
    """
    model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
    vectors = []
    batch = []
    for row in data.itertuples():
        summary = row.title + ". " + row.summary
        batch.append(summary)
        if len(batch) >= BATCH_SIZE:
            # text to vec encoding
            vectors.append(model.encode(batch))
            batch = []

    if len(batch) > 0:
        vectors.append(model.encode(batch))
        batch = []

    vectors = np.concatenate(vectors)
    return vectors


def init_search_index():
    """Function to initialize the search and upload data as collection via qdrant_client 
    """
    # get and prepare data
    prepare_data()

    # init qdrant client
    qd_client = QdrantClient(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
    )

    vec_path = os.path.join(DATA_DIR, "summary_vectors.npy")
    vecs = np.load(vec_path)
    vec_size = vecs.shape[1]

    # create payload for collection
    payload_path = os.path.join(DATA_DIR, "papers.json")
    with open(payload_path, 'r') as file:
        payload = json.load(file)

    # create collection, if exists recreate
    qd_client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=vec_size, distance="Cosine")
    )

    # upload the collection
    qd_client.upload_collection(
        collection_name=COLLECTION_NAME,
        vectors=vecs,
        payload=payload,
        ids=None,
        batch_size=64,
        parallel=1
    )


if __name__ == "__main__":
    init_search_index()
