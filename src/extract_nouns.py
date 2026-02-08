# This file is used to extract all nouns in text and 
# filter potential candidates based on similarity to taxonomy concepts

from tqdm import tqdm
from collections import defaultdict
import numpy as np
import copy
import torch
import spacy
import argparse
import os

from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

from .utils.basic_utils import *
from .utils.consts import *
from .embedding_model import NVEmbedV2EmbeddingModel


class Retriever:
    def __init__(self, report, use_gpu=True):
        """
        Initializes an instance of the class and its related components.

        Attributes
            use_batching (bool): Variable to set if batch processing.
            batch_size (int): The batch size used for processing.
            threshold (int): Similarity threshold to filter extracted nouns.
        
        Parameters:
            report (str): Name of the report document to process.
        """
        print("Initializing retriever...")
        self.use_batching = True
        self.batch_size = 32
        self.threshold = 45 #0.5
        # self.use_caching = True

        self.skip_words = SKIP_WORDS
        self.skip_chars = SKIP_CHARS
        self.skip_phrases = set(LABELS_DICT["label_mapper"].keys())
        self.skip_phrases.add("dataset")

        self.embed_model = NVEmbedV2EmbeddingModel(precomputed_embeddings_path="data/ifrs_sds_taxonomy_NV-Embed-v2")

        self.spacy_pipe = spacy.load("en_core_web_sm")
        self.TAX = load_json_file(PATH["TAX"])
        self.report_name = report

        self.RETRIEVED = {}
        
        if os.path.exists(PATH["RAG"]["prev_retrieved"]+self.report_name+".json"):
            print("Loading previous retrieved...")
            all_retrieved = load_json_file(PATH["RAG"]["prev_retrieved"]+self.report_name+".json")
            for paragraph_id, entity_dict in all_retrieved.items():
                for entity_name, entity_values in entity_dict.items():
                    self.RETRIEVED[entity_name] = entity_values

    def extract_noun_phrases(self, sentence):
        doc = self.spacy_pipe(sentence)
        noun_phrases = [chunk.text for chunk in doc.noun_chunks]
        return noun_phrases

    def filter_noun_phrases(self, noun_phrases):
        noun_phrases2 = set()

        for p in noun_phrases:
            p = p.replace("\n", "")
            skip = False
            # skip if p contains unicode string
            if any(ord(char) > 128 for char in p):
                continue
            
            p = p.strip()
            while len(p) > 1 and (not p[0].isalnum() or not p[-1].isalnum()):
                # remove starting non-characters and non-digits
                if not p[0].isalpha() and not p[0].isdigit():
                    p = p[1:]
                # remove ending non-characters and non-digits
                elif not p[-1].isalpha() and not p[-1].isdigit():
                    p = p[:-1]
                else:
                    break
            if len(p) <= 3:
                continue

            p_lower = p.lower()
            for w in self.skip_words:
                if p_lower.startswith(w + " "):
                    p = p[len(w) + 1 :]
                    p_lower = p.lower()
                    break

            if any(c in p_lower for c in self.skip_chars):
                continue
            if p_lower in self.skip_words:
                continue
            if p_lower in ["models", "number", "study", "phase", "people", 
                            "need", "what", "which", "year"]:
                continue
            if len(p) > 3:
                noun_phrases2.add(p)

        return noun_phrases2

    def retrieve_by_def(self, entity_name, entity_def):
        text = f"Name: {entity_name}\nDefinition: {entity_def}"
        node = self.embed_model.retrieve([text])[0][0]
        return node['metadata']["uuid"], node['score']

    def run(self, text):
        noun_phrases = self.extract_noun_phrases(text)
        filtered_noun_phrases = self.filter_noun_phrases(noun_phrases)

        row = {}
        phrase_i = 0
        for phrase in filtered_noun_phrases:
            phrase_i += 1
            
            if phrase in self.skip_words:
                continue
            if phrase in self.RETRIEVED:
                if self.RETRIEVED[phrase]:
                    row[phrase] = self.RETRIEVED[phrase]
            else:
                # retrieve with the retriever
                node = self.embed_model.retrieve([phrase])[0][0]

                # filter results
                if node['score'] > self.threshold:
                    uuid = node['metadata']["uuid"]
                    row[phrase] = [
                        "variable",
                        uuid,
                        self.TAX[uuid]["prefLabel"],
                        node['score'],
                    ]
                    self.RETRIEVED[phrase] = row[phrase]
                else:
                    self.RETRIEVED[phrase] = []
                    row[phrase] = []

        return row

    def save_retrieved(self):
        new = copy.deepcopy(self.RETRIEVED)
        if os.path.exists(PATH["RAG"]["prev_retrieved"]+f"{self.report_name}.json"):
            old = load_json_file(PATH["RAG"]["prev_retrieved"]+f"{self.report_name}.json")
        else:
            old = {}

        old.update(new)
        self.RETRIEVED = old
        with open(f"outputs/retrieved/{self.report_name}.json", "w") as f:
            json.dump(self.RETRIEVED, f, indent=4)


def extract(report_name):
    print("\n=== Experiment INFO ===")
    print("[INFO] Task: Noun Extraction")
    print("[INFO] Report: ", report_name)

    with open(PATH["weakly_supervised"]['path']+report_name+"/corpus.json", "r") as f:
        data = json.load(f)
    print(f"Corpus {report_name} with {len(data)} paragraphs loaded")

    retriever = Retriever(report=report_name, )

    print("[INFO] Retrieving noun phrases")
    all_retrieved = {}
    for doc in tqdm(data):
        text = doc['text']
        output = retriever.run(text)
        all_retrieved[doc['idx']] = output
    
    if os.path.exists(PATH["RAG"]["prev_retrieved"]+f"{report_name}.json"):
        old = load_json_file(PATH["RAG"]["prev_retrieved"]+f"{report_name}.json")
    else:
        old = {}

    for par_id, ent_dict in all_retrieved.items():
        if par_id not in old:
            old[par_id] = {}
        for ent_name, ent_val in ent_dict.items():
            if ent_name not in old[par_id]:
                old[par_id][ent_name] = ent_val

    with open(f"outputs/retrieved/{report_name}.json", "w") as f:
        json.dump(old, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', type=str, default='')
    args = parser.parse_args()
    report_name = args.report

    extract(report_name)
