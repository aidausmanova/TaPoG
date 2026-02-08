import os
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict, List, Tuple, Any, Optional
import logging
from copy import deepcopy
import networkx as nx

from ..embedding_model.NVEmbedV2 import NVEmbedV2EmbeddingModel

logger = logging.getLogger(__name__)

class EmbeddingStore:
    def __init__(self, embedding_model, db_filename, batch_size, namespace):
        """
        Initializes the class with necessary configurations and sets up the working directory.

        Parameters:
        embedding_model: The model used for embeddings.
        db_filename: The directory path where data will be stored or retrieved.
        batch_size: The batch size used for processing.
        namespace: A unique identifier for data segregation.

        Functionality:
        - Assigns the provided parameters to instance variables.
        - Checks if the directory specified by `db_filename` exists.
          - If not, creates the directory and logs the operation.
        - Constructs the filename for storing data in a parquet file format.
        - Calls the method `_load_data()` to initialize the data loading process.
        """
        self.embedding_model = embedding_model
        self.batch_size = batch_size
        self.namespace = namespace

        if self.namespace == "taxonomy":
            self.filename = "data/ifrs_sds_taxonomy_embeddings/vdb_taxonomy.parquet"
        else:
            if not os.path.exists(db_filename):
                logger.info(f"Creating working directory: {db_filename}")
                os.makedirs(db_filename, exist_ok=True)
            self.filename = os.path.join(
                db_filename, f"vdb_{self.namespace}.parquet"
            )
        self._load_data()

    def get_missing_string_hash_ids(self, texts):
        nodes_dict = {}

        for uid, text in texts:
            nodes_dict[f"{self.namespace}-{uid}"] = {'content': text}

        # Get all hash_ids from the input dictionary.
        all_hash_ids = list(nodes_dict.keys())
        if not all_hash_ids:
            return  {}

        existing = self.hash_id_to_row.keys()

        # Filter out the missing hash_ids.
        missing_ids = [hash_id for hash_id in all_hash_ids if hash_id not in existing]
        texts_to_encode = [nodes_dict[hash_id]["content"] for hash_id in missing_ids]

        return {h: {"hash_id": h, "content": t} for h, t in zip(missing_ids, texts_to_encode)}

    def insert_strings(self, data):
        """
        Function to insert textual information into embedding store.
        """
        nodes_dict = {}
        # all_hash_ids = []
        # {uuid: content, ...}
        for uid, text in data:
            nodes_dict[f"{self.namespace}_{uid}"] = {'content': text.lower()}
        # Get all uuids from the input dictionary.
        all_hash_ids = list(nodes_dict.keys())
        if not all_hash_ids:
            return  # Nothing to insert.

        existing = self.hash_id_to_row.keys()

        # Filter out the missing uuids.
        missing_ids = [hash_id for hash_id in all_hash_ids if hash_id not in existing]

        print(
            f"[EMB STORE] Inserting {len(missing_ids)} new records, {len(all_hash_ids) - len(missing_ids)} records already exist.")

        if not missing_ids:
            return  {}# All records already exist.

        # Prepare the texts to encode from the "content" field.
        texts_to_encode = [nodes_dict[hash_id]["content"] for hash_id in missing_ids]

        missing_embeddings = self.embedding_model.batch_encode(texts_to_encode)

        self._upsert(missing_ids, texts_to_encode, missing_embeddings)

    def _load_data(self):
        if os.path.exists(self.filename):
            df = pd.read_parquet(self.filename)
            self.hash_ids, self.texts, self.embeddings = df["hash_id"].values.tolist(), df["content"].values.tolist(), df["embedding"].values.tolist()
            self.hash_id_to_idx = {h: idx for idx, h in enumerate(self.hash_ids)}
            self.hash_id_to_row = {
                h: {"hash_id": h, "content": t}
                for h, t in zip(self.hash_ids, self.texts)
            }
            self.hash_id_to_text = {h: self.texts[idx] for idx, h in enumerate(self.hash_ids)}
            self.text_to_hash_id = {self.texts[idx]: h  for idx, h in enumerate(self.hash_ids)}
            assert len(self.hash_ids) == len(self.texts) == len(self.embeddings)
            logger.info(f"Loaded {len(self.hash_ids)} records from {self.filename}")
        else:
            self.hash_ids, self.texts, self.embeddings = [], [], []
            self.hash_id_to_idx, self.hash_id_to_row = {}, {}

    def _save_data(self):
        data_to_save = pd.DataFrame({
            "hash_id": self.hash_ids,
            "content": self.texts,
            "embedding": self.embeddings
        })
        data_to_save.to_parquet(self.filename, index=False)
        self.hash_id_to_row = {h: {"hash_id": h, "content": t} for h, t, e in zip(self.hash_ids, self.texts, self.embeddings)}
        self.hash_id_to_idx = {h: idx for idx, h in enumerate(self.hash_ids)}
        self.hash_id_to_text = {h: self.texts[idx] for idx, h in enumerate(self.hash_ids)}
        self.text_to_hash_id = {self.texts[idx]: h for idx, h in enumerate(self.hash_ids)}
        logger.info(f"Saved {len(self.hash_ids)} records to {self.filename}")

    def _upsert(self, hash_ids, texts, embeddings):
        self.embeddings.extend(embeddings)
        self.hash_ids.extend(hash_ids)
        self.texts.extend(texts)

        logger.info(f"Saving new records.")
        self._save_data()

    def get_row(self, hash_id):
        return self.hash_id_to_row[hash_id]
    
    def get_rows(self, hash_ids, dtype=np.float32):
        if not hash_ids:
            return {}

        results = {id : self.hash_id_to_row[id] for id in hash_ids}

        return results

    def get_all_ids(self):
        return deepcopy(self.hash_ids)

    def get_text_for_all_rows(self):
        return deepcopy(self.hash_id_to_row)

    def get_embedding(self, hash_id, dtype=np.float32) -> np.ndarray:
        return self.embeddings[self.hash_id_to_idx[hash_id]].astype(dtype)
    
    def get_embeddings(self, hash_ids, dtype=np.float32) -> list[np.ndarray]:
        if not hash_ids:
            return []

        indices = np.array([self.hash_id_to_idx[h] for h in hash_ids], dtype=np.intp)
        embeddings = np.array(self.embeddings, dtype=dtype)[indices]

        return embeddings
    
    def get_embedding_idx(self, vetex_key):
        return self.hash_id_to_idx.get(vetex_key, -1)


def retrieve_knn(query_ids: List[str], key_ids: List[str], query_vecs, key_vecs, k=2047, query_batch_size=1000,
                 key_batch_size=10000):
    """
    Retrieve the top-k nearest neighbors for each query id from the key ids.
    Args:
        query_ids:
        key_ids:
        k: top-k
        query_batch_size:
        key_batch_size:

    Returns:

    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if len(key_vecs) == 0: return {}

    query_vecs = torch.tensor(query_vecs, dtype=torch.float32)
    query_vecs = torch.nn.functional.normalize(query_vecs, dim=1)

    key_vecs = torch.tensor(key_vecs, dtype=torch.float32)
    key_vecs = torch.nn.functional.normalize(key_vecs, dim=1)

    results = {}

    def get_batches(vecs, batch_size):
        for i in range(0, len(vecs), batch_size):
            yield vecs[i:i + batch_size], i

    for query_batch, query_batch_start_idx in tqdm(
            get_batches(vecs=query_vecs, batch_size=query_batch_size),
            total=(len(query_vecs) + query_batch_size - 1) // query_batch_size,  # Calculate total batches
            desc="KNN for Queries"
    ):
        query_batch = query_batch.clone().detach()
        query_batch = query_batch.to(device)

        batch_topk_sim_scores = []
        batch_topk_indices = []

        offset_keys = 0

        for key_batch, key_batch_start_idx in get_batches(vecs=key_vecs, batch_size=key_batch_size):
            key_batch = key_batch.to(device)
            actual_key_batch_size = key_batch.size(0)

            similarity = torch.mm(query_batch, key_batch.T)

            topk_sim_scores, topk_indices = torch.topk(similarity, min(k, actual_key_batch_size), dim=1, largest=True,
                                                       sorted=True)

            topk_indices += offset_keys

            batch_topk_sim_scores.append(topk_sim_scores)
            batch_topk_indices.append(topk_indices)

            del similarity
            key_batch = key_batch.cpu()
            torch.cuda.empty_cache()

            offset_keys += actual_key_batch_size
        # end for each kb batch

        batch_topk_sim_scores = torch.cat(batch_topk_sim_scores, dim=1)
        batch_topk_indices = torch.cat(batch_topk_indices, dim=1)

        final_topk_sim_scores, final_topk_indices = torch.topk(batch_topk_sim_scores,
                                                               min(k, batch_topk_sim_scores.size(1)), dim=1,
                                                               largest=True, sorted=True)
        final_topk_indices = final_topk_indices.cpu()
        final_topk_sim_scores = final_topk_sim_scores.cpu()

        for i in range(final_topk_indices.size(0)):
            query_relative_idx = query_batch_start_idx + i
            query_idx = query_ids[query_relative_idx]

            final_topk_indices_i = final_topk_indices[i]
            final_topk_sim_scores_i = final_topk_sim_scores[i]

            query_to_topk_key_relative_ids = batch_topk_indices[i][final_topk_indices_i]
            query_to_topk_key_ids = [key_ids[idx] for idx in query_to_topk_key_relative_ids.cpu().numpy()]
            results[query_idx] = (query_to_topk_key_ids, final_topk_sim_scores_i.numpy().tolist())

        query_batch = query_batch.cpu()
        torch.cuda.empty_cache()
    # end for each query batch

    return results


if __name__ == "__main__":
    embedding_model = NVEmbedV2EmbeddingModel(batch_size=8)
    taxonomy_embedding_store = EmbeddingStore(embedding_model, "outputs/graph/taxonomy_embeddings", 
                                              embedding_model.batch_size, 'taxonomy')

    with open("data/ifrs_sds_taxonomy.json", "r") as f:
            taxonomy_data = json.load(f)

    taxonomy_tuples = []
    for concept_uuid, concept_data in taxonomy_data.items():
        taxonomy_tuples.append((concept_uuid, f"Label: {concept_data['prefLabel']}\nDefinition:{concept_data['ifrs_definition']}"))
    taxonomy_embedding_store.insert_strings(taxonomy_tuples)
