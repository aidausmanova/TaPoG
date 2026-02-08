from copy import deepcopy
from typing import Optional, List, Dict, Union
from pathlib import Path

import json
import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F
from transformers import AutoModel
from dataclasses import dataclass, asdict
from sentence_transformers import SentenceTransformer

from ..utils.basic_utils import get_logger

logger = get_logger(__name__)


class NVEmbedV2EmbeddingModel:
    def __init__(self, embedding_model_name: str = "nvidia/NV-Embed-v2", batch_size: int = 16, precomputed_embeddings_path: Optional[str] = None) -> None:
        """
        Initializes an instance of the class and its related components.

        Attributes
            embedding_model (AutoModel): Specified transfomers AutoModel.
            max_length (int): 
            batch_size (int): The batch size used for processing.
        
        Parameters
            embedding_model_name (str): Name of the embedding model in HuggingFace.
            batch_size (int): The batch size used for processing.
            precomputed_embeddings_path (str): Path to the precomputed embeddings.
        """
        if embedding_model_name == "BAAI/bge-large-en-v1.5" or embedding_model_name == "sentence-transformers/all-mpnet-base-v2":
            self.embedding_model = SentenceTransformer(embedding_model_name)
            self.max_length = 384 #512
        else:
            self.embedding_model = AutoModel.from_pretrained(embedding_model_name, trust_remote_code=True, torch_dtype=torch.float16, device_map = 'auto')
            self.max_length = 32768

        self.embedding_model.eval()
        self.model_name = embedding_model_name
        self.batch_size = batch_size
        self.taxonomy_embeddings = {}
        self.taxonomy_metadata = {}
        self.taxonomy_embedding_matrix = None

        if precomputed_embeddings_path:
            self._load_precomputed_entities(precomputed_embeddings_path)

    def _load_precomputed_entities(self, path: str) -> tuple:
        """
        Function to load already precomputed vector embeddings.
        """
        embeddings = np.load(path+"/vector_store.npy")
        metadata_path = path+"/metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        self.taxonomy_embeddings = embeddings
        self.taxonomy_embedding_matrix = torch.from_numpy(embeddings)
        self.taxonomy_metadata = metadata

    def batch_encode(self, texts: List[str], 
                     instruction: Optional[str] = "",
                     metadata: Optional[List[Dict]] = None,
                     save_embeddings: bool = False,
                     save_path: str = "data/") -> None:
        if isinstance(texts, str): texts = [texts]
        
        #### Generate embeddings
        if self.model_name == "BAAI/bge-large-en-v1.5" or self.model_name == "sentence-transformers/all-mpnet-base-v2":
            if instruction:
                texts = [instruction + text for text in texts]
            
            # Generate embeddings
            results = self.embedding_model.encode(
                texts, 
                batch_size=self.batch_size,
                show_progress_bar=True,
                normalize_embeddings=True
            )
        else:
            if len(texts) <= self.batch_size:
                results = self.embedding_model.encode(texts, instruction=instruction, max_length=self.max_length, batch_size=self.batch_size)
            else:
                pbar = tqdm(total=len(texts), desc="Batch Encoding")
                results = []
                for i in range(0, len(texts), self.batch_size):
                    prompts = texts[i:i + self.batch_size]
                    results.append(self.embedding_model.encode(prompts, instruction=instruction, max_length=self.max_length, batch_size=len(prompts)))
                    pbar.update(self.batch_size)
                pbar.close()
                results = torch.cat(results, dim=0)

            if isinstance(results, torch.Tensor):
                results = F.normalize(results, p=2, dim=1)
                results = results.cpu()
                results = results.numpy()
            else:
                results = (results.T / np.linalg.norm(results, axis=1)).T

        if metadata is None:
            return results

        if save_embeddings:
            np.save(save_path+"/vector_store.npy", results)
            with open(save_path+"/metadata.json", 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        return {
            'embeddings': results,
            'metadata': metadata,
            'texts': texts
        }

    def retrieve(self, texts: List[str], top_k=5):
        """
        Function to retreive an embedding closest to the query embedding.
        """
        task_name_to_instruct = {"example": "Given a noun term, retrieve entity that is ",}
        query_prefix = "Instruct: "+task_name_to_instruct["example"]+"\nQuery: "

        query_embeddings = self.embedding_model.encode(texts, instruction=query_prefix, max_length=self.max_length)
        if isinstance(query_embeddings, torch.Tensor):
            query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
            query_embeddings = query_embeddings.cpu()
            query_embeddings = query_embeddings.numpy()
        else:
            query_embeddings = (query_embeddings.T / np.linalg.norm(query_embeddings, axis=1)).T

        scores = (query_embeddings @ self.taxonomy_embeddings.T) * 100
        single_query = len(scores.shape) == 1
    
        if single_query:
            scores = scores.reshape(1, -1)

        all_results = []
        for query_scores in scores:
            top_indices = np.argsort(query_scores)[::-1][:top_k]
            top_scores = query_scores[top_indices]

            results = []
            for rank, (idx, score) in enumerate(zip(top_indices, top_scores), 1):
                metadata = self.taxonomy_metadata[idx]
                result = {
                    'rank': rank,
                    'index': int(idx),
                    'score': float(score),
                    'metadata': metadata
                }
                results.append(result)
            
            all_results.append(results)

        if single_query:
            return all_results[0]
        return all_results
