# This file is used to create precompouted embeddings for taxonomy concepts

import os
import json
import torch
import argparse
import numpy as np
from typing import List, Optional
from pathlib import Path

from transformers import AutoModel, AutoTokenizer
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.core.embeddings import BaseEmbedding

from .NVEmbedV2 import NVEmbedV2EmbeddingModel


def get_taxonomy_corpus(taxonomy_path, is_llama_doc=True, include_related_terms=False):
    with open(taxonomy_path+".json", "r") as f:
        taxonomy_data = json.load(f)
    
    taxonomy_documents = []
    metadata = []
    for uid, taxonomy_metadata in taxonomy_data.items():
        if is_llama_doc:
            taxonomy_documents.append(
                Document(
                    text = f"Label: {taxonomy_metadata['prefLabel']}\n Definition: {taxonomy_metadata['enriched_definition']}\n Related terms: {related_terms}",
                    metadata = {
                        "uuid": uid,
                        "label": taxonomy_metadata['prefLabel'],
                        "tags": taxonomy_metadata['tags'],
                        "definition": taxonomy_metadata['enriched_definition'],
                        "relatedTerms": taxonomy_metadata['relatedTerms'],
                        "path_id": taxonomy_metadata['path_id'],
                        "path_label": taxonomy_metadata['path_label']
                    }
                )
            )
        else:
            taxonomy_documents.append(str(f"Label: {taxonomy_metadata['prefLabel']}\n Definition: {taxonomy_metadata['enriched_definition']}")) # \n Related terms: {related_terms}
            metadata.append(
                {
                    "uuid": uid,
                    "label": taxonomy_metadata['prefLabel'],
                    "definition": taxonomy_metadata['enriched_definition'],
                    "path_id": taxonomy_metadata['path_id'],
                    "path_label": taxonomy_metadata['path_label']
                }
            )

    return taxonomy_documents, metadata

def save_embeddings_pytorch(embeddings, entities, save_dir):
    Path(save_dir).mkdir(exist_ok=True)
    
    # Convert to numpy if tensor
    if torch.is_tensor(embeddings):
        embeddings_np = embeddings.cpu().numpy()
    else:
        embeddings_np = embeddings
    
    # Save embeddings and metadata separately
    np.save(f"{save_dir}/embeddings.npy", embeddings_np)
    
    with open(f"{save_dir}/metadata.json", 'w') as f:
        json.dump({
            'entities': entities,
            'embedding_dim': embeddings_np.shape[1],
            'num_embeddings': embeddings_np.shape[0]
        }, f, indent=2)
    
    print(f"Saved to {save_dir}/embeddings.npy and metadata.json")

class NVidiaEmbedder(BaseEmbedding):
    def __init__(self, model_name="nvidia/NV-Embed-v2", device="cuda", max_length=512, add_instructions=True, normalize_embeddings=True, **kwargs):
        super().__init__(**kwargs)
        self._max_length = max_length
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
        self._device = device
        self._add_instructions = add_instructions
        self._normalize_embeddings = normalize_embeddings
        self._model.eval()

        self._passage_instruction = "Represent this climate disclosure taxonomy concept for retrieval: "
        self._query_instruction = "Represent this climate-related query for searching relevant taxonomy concepts: "


    @classmethod
    def class_name(cls) -> str:
        return "nvidia_nv_embed_v2"
    
    def _add_instruction_prefix(self, text: str, is_query: bool = False) -> str:
        """Add instruction prefix to guide the model."""
        if not self._add_instructions:
            return text
        
        instruction = self._query_instruction if is_query else self._passage_instruction
        return instruction + text
    
    def _normalize(self, embeddings: torch.Tensor) -> torch.Tensor:
        """L2 normalization for better cosine similarity."""
        if not self._normalize_embeddings:
            return embeddings
        return torch.nn.functional.normalize(embeddings, p=2, dim=1)

    def _embed(self, texts: List[str], is_query: bool = False) -> List[List[float]]:
        if self._add_instructions:
            texts = [self._add_instruction_prefix(t, is_query) for t in texts]
        inputs = self._tokenizer(texts, padding=True, 
                                 truncation=True, max_length=self._max_length,
                                 return_tensors="pt").to(self._device)
        with torch.no_grad():
            outputs = self._model(**inputs)

        embeddings = outputs['sentence_embeddings'][0]
        if self._normalize_embeddings:
            embeddings = self._normalize(embeddings)

        embeddings_list = [
            [float(x) for x in emb.detach().cpu().numpy()] 
            for emb in embeddings
        ]
        return embeddings_list

    def _get_text_embedding(self, text: str):
        return self._embed([text])[0]

    def _get_query_embedding(self, query: str):
        return self._get_text_embedding(query)

    def _get_text_embeddings(self, texts):
        return self._embed(texts)
        
    async def _aget_query_embedding(self, query: str):
        return self._get_text_embedding(query)
    
    async def _aget_text_embedding(self, text):
        return self._get_text_embedding(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--taxonomy', type=str, default='data/ifrs_sdd_taxonomy')
    parser.add_argument('--is_llama_index', type=bool, default=False)
    args = parser.parse_args()

    llama_index = args.is_llama_index
    taxonomy_path = args.taxonomy
    taxonomy_documents, taxonomy_metadata = get_taxonomy_corpus(taxonomy_path, is_llama_doc=llama_index, include_related_terms=False)
    print("[INFO] Taxonomy loaded")
    
    if llama_index:
        embed_model = NVidiaEmbedder(device="cuda")
        Settings.embed_model = embed_model

        print("[INFO] Start generating embeddings")
        index = VectorStoreIndex.from_documents(taxonomy_documents, embed_model=Settings.embed_model, show_progress=True)
        index.storage_context.persist(persist_dir="data/ifrs_NV-Embed-v2")
    else:
        save_path = taxonomy_path+"_NV-Embed-v2"
        embed_model = NVEmbedV2EmbeddingModel(embedding_model_name="nvidia/NV-Embed-v2")
        embeddings = embed_model.batch_encode(texts=taxonomy_documents, 
                                              metadata=taxonomy_metadata, 
                                              save_embeddings=True, 
                                              save_path=save_path)
    print("[INFO] Finihsed execution")
