import os
import re
import json
import uuid
import logging
import networkx as nx
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from tqdm import tqdm

from ..embedding_model.NVEmbedV2 import NVEmbedV2EmbeddingModel
from ..graph.reranking import DSPyFilter
from ..llm.meta_llama import InfoExtractor
from ..prompts.prompt_template_manager import *
from ..utils.basic_utils import *
from ..utils.consts import *
from ..utils.embedding_store import EmbeddingStore, retrieve_knn
from ..utils.eval_utils import *

logger = logging.getLogger(__name__)

@dataclass
class QuerySolution:
    question: str
    docs: List[str]
    doc_ids: List[str] = None
    doc_scores: np.ndarray = None
    answer: str = None
    gold_answers: List[str] = None
    gold_docs: Optional[List[str]] = None
    recalls: dict = None
    precisions: dict = None
    f1: dict = None
    ndcg: dict = None
    mrr: float = None
    top_triples: Optional[List[str]] = None


    def to_dict(self):
        return {
            "question": self.question,
            "answer": self.answer,
            "gold_answers": self.gold_answers,
            "recalls": self.recalls,
            "precisions": self.precisions,
            "f1": self.f1,
            "ndcg": self.ndcg,
            "mrr": self.mrr,
            "top_triples": self.top_triples,
            "doc_ids": self.doc_ids,
            "gold_docs": self.gold_docs,
            "docs": self.docs,
            "doc_scores": [round(v, 4) for v in self.doc_scores.tolist()]  if self.doc_scores is not None else None
            
        }

class ReportKnowledgeGraph:
    def __init__(self, report, taxonomy, llm: str = 'meta-llama/Llama-3.1-70B-Instruct', synonym_sim_threshold: float = 0.8, link_top_k: int = 5, passage_node_weight: float = 0.05, damping_factor: float = 0.5):
        """
        Initializes an instance of the class and its related components.

        Attributes:
            report_name (str): Name of the current processed report document.
            synonym_sim_threshold (float): Threshold value for KNN-based similarity scores.
            embedding_model (NVEmbedV2EmbeddingModel): The embedding model associated with the current configuration.
            working_dir (str): The directory where graph specific information will be stored.
        Parameters:
            report (str): Name of the current processed report document.
            synonym_sim_threshold (float): Threshold value for KNN-based similarity scores.
        """
        self.report_name = report
        self.taxonomy = taxonomy
        self.synonym_sim_threshold = synonym_sim_threshold
        self.link_top_k = link_top_k
        self.passage_node_weight = passage_node_weight
        self.damping_factor = damping_factor
        self.llm_name = llm

        self.working_dir = os.path.join(PATH['KG'], self.report_name)

        self.embedding_model = NVEmbedV2EmbeddingModel(embedding_model_name="nvidia/NV-Embed-v2", batch_size=8)
        self.llm_model = InfoExtractor()
        self.rerank_filter = DSPyFilter(self)
        self.graph = nx.DiGraph()
        
        self.entity_embeddings = {}
        self.entities = {}
        self.triples = {}
        self.entity_uuids = {}
        self.active_paragraph_ids = []
        self.paragraph_uuids = {}
        
        self.concept2concept_relations = 0
        self.entity2concept_relations = 0
        self.entity2paragraph_relations = 0
        self.entity2entity_relations = 0
        self.synonym_relations = 0

        self.graph_triples = []
        self.ent_node_to_num_chunk = {}

        self.taxonomy_embedding_store = EmbeddingStore(self.embedding_model,
                                                       os.path.join(self.working_dir, "taxonomy_embeddings"),
                                                       self.embedding_model.batch_size, 'taxonomy')
        self.paragraph_embedding_store = EmbeddingStore(self.embedding_model,
                                                        os.path.join(self.working_dir, "paragraph_embeddings"),
                                                        self.embedding_model.batch_size, 'paragraph')
        self.entity_embedding_store = EmbeddingStore(self.embedding_model,
                                                     os.path.join(self.working_dir, "entity_embeddings"),
                                                     self.embedding_model.batch_size, 'entity')
        # A fact is a triple with two entities and relationships between them
        self.fact_embedding_store = EmbeddingStore(self.embedding_model, 
                                                   os.path.join(self.working_dir, "fact_embeddings"),
                                                   self.embedding_model.batch_size, 'fact')

        print(f"[GRAPH] Graph inititialized for report {self.report_name}")
        self.build_graph()
    
    ############################# GRAPH CONSTRUCTION #############################
    def load_taxonomy_concepts(self):
        """
        Load taxonomy concepts into Embedding Store and add as nodes to graph with hierarchical relationships
        """
        with open(f"data/{self.taxonomy}.json", "r") as f:
            taxonomy_data = json.load(f)
        
        taxonomy_tuples = []
        taxonomy_metadata = {}
        for concept_uuid, concept_data in taxonomy_data.items():
            taxonomy_metadata[concept_uuid] = {
                'label': concept_data['prefLabel'], 
                'parent_concept_uuid': concept_data['path_id'][0] if len(concept_data['path_id']) > 0 else None  # First item in path_id is the immediate parent
            }
            taxonomy_tuples.append((concept_uuid, f"Label: {concept_data['prefLabel']}\nDefinition:{concept_data['enriched_definition']}"))
        
        taxonomy_node_keys = self.taxonomy_embedding_store.get_all_ids()
        if len(taxonomy_node_keys) == 0:
            self.taxonomy_embedding_store.insert_strings(taxonomy_tuples)

        print("[GRAPH] Loading taxonomy concepts into graph ...")
        for taxonomy_id in taxonomy_node_keys:
            concept_uuid = taxonomy_id.split("_")[1]
            self.graph.add_node(
                taxonomy_id,
                node_type='concept',
                uuid=concept_uuid,
                label=taxonomy_metadata[concept_uuid]['label']
            )

            if taxonomy_metadata[concept_uuid]['parent_concept_uuid']:
                self.graph.add_edge(
                    f"taxonomy_{concept_uuid}",
                    f"taxonomy_{taxonomy_metadata[concept_uuid]['parent_concept_uuid']}",
                    weight = 1.0, 
                    edge_type = "hierarchical", 
                    relationship = "is subtopic of"
                )
                self.concept2concept_relations += 1
        print(f"[GRAPH] Loaded {len(taxonomy_data)} concept nodes into graph, with {self.concept2concept_relations} relations")

    def load_paragraphs(self):
        """
        Load paragraphs into Embedding Store and as nodes to graph
        """
        print("[GRAPH] Loading paragraph chunks into embedding store ...")
        with open(PATH['weakly_supervised']['path']+self.report_name+"/corpus.json", "r") as f:
            paragraphs_data = json.load(f)
        
        paragraph_data = {}
        paragraph_tuples = []
        for paragraph in paragraphs_data:
            paragraph_data[paragraph['uid']] = {
                'idx': paragraph['idx']
            }
            self.paragraph_uuids[paragraph['idx']] = paragraph['uid']
            paragraph_tuples.append((paragraph['uid'], paragraph['text']))

        paragraph_node_keys = list(self.paragraph_embedding_store.get_all_ids())
        if len(paragraph_node_keys) == 0:
            self.paragraph_embedding_store.insert_strings(paragraph_tuples)

        ############ Need to change that to retrieve from embeddings store ############
        paragraph_node_keys = list(self.paragraph_embedding_store.get_all_ids())
        print("[GRAPH] Loading paragraph chunks into graph ...")
        for paragraph_id in paragraph_node_keys:
            uid = paragraph_id.split("_")[1]
            self.graph.add_node(
                    paragraph_id,
                    node_type='paragraph',
                    uuid=uid,
                    label=paragraph_data[uid]['idx']
                )
        print(f"[GRAPH] Loaded {len(self.paragraph_uuids)} paragraph nodes into graph")

    def preprocess_extracted_data(self):
        """
        Function to get extracted entities and their relationships.
        """
        with open(PATH['RAG']['post_retrieved']+self.report_name+".json", "r") as f:
            postrag_data = json.load(f)
        
        entity_tuples = []
        self.entity_uuids = postrag_data['entity_id_mapping']
        # Load entities
        for entity_data in postrag_data['entities']:
            entity = {
                    "label": entity_data['canonical_name'],
                    "entity_type": entity_data['label'],
                    "definition": entity_data['definition'],
                    "paragraph_ids": entity_data['mentioned_in_paragraphs'],
                    "taxonomy_concepts": [(entity_data['taxonomy_uuid'], entity_data['score'])] if 'taxonomy_uuid' in entity_data else []
                }
            self.entities[entity_data['id']] = entity
            entity_tuples.append((entity_data['id'], f"{entity_data['canonical_name']}\n{entity_data['definition']}"))

        # Load relationship data
        for relation in postrag_data['relationships']:
            self.triples[f"{relation['source_id']}_{relation['target_id']}"] = relation
            self.add_fact_edge(relation['source_id'],relation['target_id'], relation['relation'])

        if len(self.entity_embedding_store.get_all_ids()) == 0:
            self.entity_embedding_store.insert_strings(entity_tuples)

    def load_entities(self):
        """
        Load entities into graph and add relationships with paragraphs that mention entity and taxonomy concept closest to the entity.
        """
        print("[GRAPH] Loading extracted entities into graph ...")
        entity_node_keys: List = list(self.entity_embedding_store.get_all_ids())
        for entity_id in entity_node_keys:
            entity_uuid = entity_id.split("_")[1]
            self.graph.add_node(
                entity_id,
                node_type='entity',
                uuid=entity_uuid,
                label=self.entities[entity_uuid]['label'],
                entity_type=self.entities[entity_uuid]['entity_type'],
                definition=self.entities[entity_uuid]['definition'],
                paragraphs=",".join(str(element) for element in self.entities[entity_uuid]['paragraph_ids']),
                concepts=",".join(str(element[0]) for element in self.entities[entity_uuid]['taxonomy_concepts']),
            )
                    
            # Add edge between entity and pargaraph
            for paragraph_idx in self.entities[entity_uuid]['paragraph_ids']:
                self.add_node_to_paragraph_edge(entity_uuid, self.paragraph_uuids[paragraph_idx])
                self.entity2paragraph_relations += 1
            
            # Add edge between entity and taxonomy concept
            for taxonomy_uuid, taxonomy_sim_score in self.entities[entity_uuid]['taxonomy_concepts']:
                self.add_node_to_concept_edge(entity_uuid, taxonomy_uuid, taxonomy_sim_score/100)
                self.entity2concept_relations += 1
                self.triples[f"{entity_uuid}_{taxonomy_uuid}"] = {
                    "source_id": entity_uuid,
                    "source_name": self.entities[entity_uuid]['label'],
                    "relation": "is linked to",
                    "target_id": taxonomy_uuid,
                    "target_name": self.taxonomy_embedding_store.get_text_for_all_rows()[f"taxonomy_{taxonomy_uuid}"]['content'].split("\n")[0]
                }
            
        print(f"[GRAPH] Loaded {len(entity_node_keys)} entity nodes into graph")

    def connect_synonyms(self):
        """
        KNN-based synonym entity retrieval.
        """
        self.entity_id_to_row = self.entity_embedding_store.get_text_for_all_rows()
        entity_node_keys = list(self.entity_id_to_row.keys())

        logger.info(f"Performing KNN retrieval for each entity nodes ({len(entity_node_keys)}).")

        entity_embs = self.entity_embedding_store.get_embeddings(entity_node_keys)

        query_node_key2knn_node_keys = retrieve_knn(query_ids=entity_node_keys,
                                                    key_ids=entity_node_keys,
                                                    query_vecs=entity_embs,
                                                    key_vecs=entity_embs,
                                                    k=2047,
                                                    query_batch_size=1000,
                                                    key_batch_size=10000)
        num_synonym_triple = 0
        synonym_candidates = []  # [(node key, [(synonym node key, corresponding score), ...]), ...]

        for curr_node_key in tqdm(query_node_key2knn_node_keys.keys(), total=len(query_node_key2knn_node_keys)):
            synonyms = []
            entity = self.entity_id_to_row[curr_node_key]["content"]

            if len(re.sub('[^A-Za-z0-9]', '', entity)) > 2:
                synonym_nodes = query_node_key2knn_node_keys[curr_node_key]

                num_nns = 0
                for synonym_node_key, synonym_node_score in zip(synonym_nodes[0], synonym_nodes[1]):
                    if synonym_node_score < self.synonym_sim_threshold or num_nns > 100:
                        break

                    synonym_entity = self.entity_id_to_row[synonym_node_key]["content"]

                    if synonym_node_key != curr_node_key and synonym_entity != '':
                        synonyms.append((synonym_node_key, synonym_node_score))
                        num_synonym_triple += 1
                        num_nns += 1

            synonym_candidates.append((curr_node_key, synonyms))

        # Add synonym edges to graph
        new_syn_edges = 0
        for curr_node_key, syn_nodes in synonym_candidates:
            for curr_syn_node_key, curr_syn_score in syn_nodes:
                self.graph.add_edge(
                    curr_node_key,
                    curr_syn_node_key,
                    weight = curr_syn_score, 
                    edge_type = "synonym", 
                    relationship = "similar_to"
                )
                # Add synonymity edges both ways
                self.graph.add_edge(
                    curr_syn_node_key,
                    curr_node_key,
                    weight = curr_syn_score, 
                    edge_type = "synonym", 
                    relationship = "similar_to"
                )
                new_syn_edges += 2
        self.synonym_relations += new_syn_edges
        print(f"[GRAPH] Added {new_syn_edges} new synonym edges")

    def build_graph(self):
        """
        Build the complete knowledge graph for a given report
        """
        graph_file = os.path.join(self.working_dir, f"graph.graphml")
        
        with open(PATH['RAG']['post_retrieved']+self.report_name+".json", "r") as f:
            postrag_data = json.load(f)
        
        for row in postrag_data['relationships']:
            s = f"entity_{row['source_id']}"
            t = f"entity_{row['target_id']}"
            self.ent_node_to_num_chunk[s] = self.ent_node_to_num_chunk.get(s, 0) + 1
            self.ent_node_to_num_chunk[t] = self.ent_node_to_num_chunk.get(t, 0) + 1

        for row in postrag_data['entities']:
            if 'taxonomy_uuid' in row:
                s = f"entity_{row['id']}"
                t = f"taxonomy_{row['taxonomy_uuid']}"
                self.ent_node_to_num_chunk[s] = self.ent_node_to_num_chunk.get(s, 0) + 1
                self.ent_node_to_num_chunk[t] = self.ent_node_to_num_chunk.get(t, 0) + 1

        if os.path.exists(graph_file):
            print("[GRAPH] Graph file exists")
            self.load_graph(graph_file)
        else:
            # Load data into embedding stor and create nodes
            self.load_taxonomy_concepts()
            self.preprocess_extracted_data()
            self.load_paragraphs()
            self.load_entities()

            triple_tuples = [(triple_id, f"('{triple['source_name']}', '{triple['relation']}', '{triple['target_name']}')") for triple_id, triple in self.triples.items()]
            if len(self.fact_embedding_store.get_all_ids()) == 0:
                self.fact_embedding_store.insert_strings(triple_tuples)

            print("[GRAPH] All nodes loaded. Start synonymity detection ...")
            self.connect_synonyms()

            self.save_graph(self.working_dir)
            print("[GRAPH] Graph construction completed!")
        print(self.get_graph_info())
        
        # return self.graph
    
    def load_graph(self, graph_file):
        self.graph = nx.read_graphml(graph_file)
        print(f"Loaded {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")

    def save_graph(self, output_path: str, format: str = 'graphml') -> None:
        """
        Save graph to graphml file.
        """
        print(f"Writing graph with {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
        graph_output_path = os.path.join(self.working_dir, f"graph.graphml")
        if format == 'graphml':
            nx.write_graphml(self.graph, graph_output_path)
        elif format == 'gexf':
            nx.write_gexf(self.graph, output_path)
        elif format == 'json':
            from networkx.readwrite import json_graph
            graph_data = json_graph.node_link_data(self.graph)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(graph_data, f, indent=2)
        elif format == 'pickle':
            nx.write_gpickle(self.graph, output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"Graph saved to {output_path}")
    
    def get_graph_info(self):
        """
        Function to get statistical information about constructed graph.
        """
        graph_GRAPH = {}
        entity_nodes_keys = self.entity_embedding_store.get_all_ids()
        graph_GRAPH["num_entity_nodes"] = len(entity_nodes_keys)
        paragraph_nodes_keys = self.paragraph_embedding_store.get_all_ids()
        graph_GRAPH["num_paragraph_nodes"] = len(paragraph_nodes_keys)
        concept_nodes_keys = self.taxonomy_embedding_store.get_all_ids()
        graph_GRAPH["num_concept_nodes"] = len(set(concept_nodes_keys))
        graph_GRAPH["num_total_nodes"] = graph_GRAPH["num_entity_nodes"] + graph_GRAPH["num_paragraph_nodes"] + graph_GRAPH["num_concept_nodes"]

        graph_GRAPH['nun_hierarchical_edges'] = self.concept2concept_relations
        graph_GRAPH['nun_paragraph_mention_edges'] = self.entity2paragraph_relations
        graph_GRAPH['num_entity_link_edges'] = self.entity2concept_relations
        graph_GRAPH['num_fact_edges'] = self.entity2entity_relations
        graph_GRAPH['num_synonym_edges'] = self.synonym_relations
        graph_GRAPH['num_total_edges'] = self.concept2concept_relations + self.entity2paragraph_relations + self.entity2concept_relations + self.entity2entity_relations + self.synonym_relations
        return graph_GRAPH

    def add_node_to_paragraph_edge(self, entity_uuid, paragraph_uuid, relation="is mentioned in"):
        """
        Function to add edge between entity and its paragraph.
        """
        self.graph.add_edge(
                    f"entity_{entity_uuid}",
                    f"paragraph_{paragraph_uuid}",
                    weight = 1.0, 
                    edge_type = "mention", 
                    relationship = relation
                )

    def add_node_to_concept_edge(self, entity_uuid, concept_uuid, score=None, relation="is linked to"):
        """
        Function to add edge between entity and taxonomy concept.
        """
        self.graph.add_edge(
                    f"entity_{entity_uuid}",
                    f"taxonomy_{concept_uuid}",
                    weight = score, 
                    edge_type = "link", 
                    relationship = relation
                )
    
    def add_fact_edge(self, e1_uuid, e2_uuid, relation):
        """
        Function to add edge between two entities.
        """
        logger.info("[GRAPH] Adding fact edges ...")
        self.graph.add_edge(
            f"entity_{e1_uuid}",
            f"entity_{e2_uuid}",
            weight = 1.0, # Change to mention frequency?
            edge_type = "fact", 
            relationship = relation
        )
        self.entity2entity_relations += 1

    ############################# GRAPH RETRIEVAL #############################
    def retrieve(self,
                 queries: List[str],
                 num_to_retrieve: int = 10,
                 gold_docs: List[List[str]] = None) -> List[QuerySolution] | Tuple[List[QuerySolution], Dict]:
        """
        Performs retrieval using the HippoRAG 2 framework, which consists of several steps:
        - Fact Retrieval
        - Recognition Memory for improved fact selection
        - Dense passage scoring
        - Personalized PageRank based re-ranking

        Parameters:
            queries: List[str]
                A list of query strings for which documents are to be retrieved.
            num_to_retrieve: int, optional
                The maximum number of documents to retrieve for each query. If not specified, defaults to
                the `retrieval_top_k` value defined in the global configuration.
            gold_docs: List[List[str]], optional
                A list of lists containing gold-standard documents corresponding to each query. Required
                if retrieval performance evaluation is enabled (`do_eval_retrieval` in global configuration).

        Returns:
            List[QuerySolution] or (List[QuerySolution], Dict)
                If retrieval performance evaluation is not enabled, returns a list of QuerySolution objects, each containing
                the retrieved documents and their scores for the corresponding query. If evaluation is enabled, also returns
                a dictionary containing the evaluation metrics computed over the retrieved results.

        Notes
        -----
        - Long queries with no relevant facts after reranking will default to results from dense passage retrieval.
        """
        
        self.prepare_retrieval_objects()
        self.get_query_embeddings(queries)

        retrieval_results = []
        retrieval_results_dict = []
        top_triples = []
        for q_idx, query in tqdm(enumerate(queries), desc="Retrieving", total=len(queries)):
            print("#######################################")
            print("[QUERY]: ", query)
            query_fact_scores = self.get_fact_scores(query)
            top_k_fact_indices, top_k_facts, rerank_log = self.rerank_facts(query, query_fact_scores)
            print("[RERANK LOG]", rerank_log)
            if len(top_k_facts) == 0:
                logger.info('No facts found after reranking, return DPR results')
                print("No facts found after reraning")
                top_triples.append(None)
                sorted_doc_ids, sorted_doc_scores = self.dense_passage_retrieval(query)
            else:
                print("[TOP K FACTS]: ", top_k_facts)
                top_triples.append(top_k_facts)
                sorted_doc_ids, sorted_doc_scores = self.graph_search_with_fact_entities(query=query,
                                                                                         link_top_k=self.link_top_k,
                                                                                         query_fact_scores=query_fact_scores,
                                                                                         top_k_facts=top_k_facts,
                                                                                         top_k_fact_indices=top_k_fact_indices,
                                                                                         passage_node_weight=self.passage_node_weight)
            
            # Get top paragraphs
            top_k_docs = [self.paragraph_embedding_store.get_row(self.paragraph_node_keys[idx])["content"] for idx in sorted_doc_ids[:num_to_retrieve]]
            tok_k_doc_ids = [self.graph.nodes[self.paragraph_node_keys[idx]]['label'] for idx in sorted_doc_ids[:num_to_retrieve]]

            query_result = QuerySolution(question=query, docs=top_k_docs, doc_ids=tok_k_doc_ids, doc_scores=sorted_doc_scores[:num_to_retrieve])
            retrieval_results.append(query_result)
            
        # Evaluate retrieval
        if gold_docs is not None:
            k_list = [1, 2, 5, 10, 15]
            retrieved_docs=[retrieval_result.docs for retrieval_result in retrieval_results]
            retrieved_doc_ids=[retrieval_result.doc_ids for retrieval_result in retrieval_results]
            overall_reecall_result, example_recall_results = calculate_recall_k(gold_docs=gold_docs, retrieved_docs=retrieved_doc_ids, k_list=k_list)
            overall_precision, example_precision = calculate_precision_k(gold_docs=gold_docs, retrieved_docs=retrieved_doc_ids, k_list=k_list)
            overall_f1, example_f1 = calculate_f1_k(gold_docs=gold_docs, retrieved_docs=retrieved_doc_ids, k_list=k_list)
            overall_ndcg, example_ndcg = calculate_ndcg_k(gold_docs=gold_docs, retrieved_docs=retrieved_doc_ids, k_list=k_list)
            overall_mrr, reciprocal_ranks = calculate_mrr(gold_docs=gold_docs, retrieved_docs=retrieved_doc_ids)
            logger.info(f"Evaluation results for retrieval: {overall_reecall_result}")
            print("REPORT: ", self.report_name)
            print(f"Recall results: {overall_reecall_result}")
            print(f"Precision results: {overall_precision}")
            print(f"F1 results: {overall_f1}")
            print(f"NDCG results: {overall_ndcg}")
            print(f"MRR results: {overall_mrr}")

            overall_retrieval_result = {
                "R@1": overall_reecall_result['Recall@1'],
                "R@5": overall_reecall_result['Recall@5'],
                "R@10": overall_reecall_result['Recall@10'],
                "R@15": overall_reecall_result['Recall@15'],
                "P@5": overall_precision['Precision@5'],
                "P@10": overall_precision['Precision@10'],
                "P@15": overall_precision['Precision@15'],
                "F1@5": overall_f1['F1@5'],
                "F1@10": overall_f1['F1@10'],
                "F1@15": overall_f1['F1@15'],
                "N@5": overall_ndcg['NDCG@5'],
                "N@10": overall_ndcg['NDCG@10'],
                "N@15": overall_ndcg['NDCG@15'],
                "MRR": overall_mrr
            }

            for i in range(len(queries)):
                retrieval_results[i].recalls = example_recall_results[i]
                retrieval_results[i].precisions = example_precision[i]
                retrieval_results[i].f1 = example_f1[i]
                retrieval_results[i].ndcg = example_ndcg[i]
                retrieval_results[i].mrr = reciprocal_ranks[i]
                retrieval_results[i].gold_docs = gold_docs[i]
                retrieval_results[i].top_triples = top_triples[i]
                retrieval_results_dict.append(retrieval_results[i].to_dict())

            with open(self.working_dir+"/retrieval_results.json", "w") as f:
                json.dump(retrieval_results_dict, f, indent=4)
            return retrieval_results, overall_retrieval_result
        else:
            return retrieval_results
        
    def prepare_retrieval_objects(self):
        """
        Prepares various in-memory objects and attributes necessary for fast retrieval processes, such as embedding data and graph relationships, ensuring consistency
        and alignment with the underlying graph structure.
        """

        logger.info("[GRAPH] Preparing for fast retrieval")
        self.query_to_embedding: Dict = {'triple': {}, 'paragraph': {}, 'conept': {}}

        self.entity_node_keys: List = list(self.entity_embedding_store.get_all_ids()) # a list of phrase node keys
        self.paragraph_node_keys: List = list(self.paragraph_embedding_store.get_all_ids()) # a list of passage node keys
        self.concept_node_keys: List = list(self.taxonomy_embedding_store.get_all_ids()) # a list of taxonomy node keys
        self.fact_node_keys: List = list(self.fact_embedding_store.get_all_ids()) # a list of fact node keys

        igraph_name_to_idx = {node: idx for idx, node in enumerate(self.graph.nodes())}

        self.node_name_to_vertex_idx = igraph_name_to_idx
        self.entity_node_idxs = [igraph_name_to_idx[node_key] for node_key in self.entity_node_keys] # a list of backbone graph entity node index
        self.paragraph_node_idxs = [igraph_name_to_idx[node_key] for node_key in self.paragraph_node_keys] # a list of backbone passage node index
        self.concept_node_idxs = [igraph_name_to_idx[node_key] for node_key in self.concept_node_keys] # a list of backbone graph concept node index

        self.paragraph_key_to_idx = {key: idx for key, idx in zip(self.paragraph_node_keys, self.paragraph_node_idxs)}
        self.concept_key_to_idx = {key: idx for key, idx in zip(self.concept_node_keys, self.concept_node_idxs)}

        self.entity_embeddings = np.array(self.entity_embedding_store.get_embeddings(self.entity_node_keys))
        self.paragraph_embeddings = np.array(self.paragraph_embedding_store.get_embeddings(self.paragraph_node_keys))
        self.concept_embeddings = np.array(self.taxonomy_embedding_store.get_embeddings(self.concept_node_keys))
        self.fact_embeddings = np.array(self.fact_embedding_store.get_embeddings(self.fact_node_keys))

        self.entity_name_keys = get_names_to_keys_dict(self.entity_embedding_store.get_text_for_all_rows())
        self.concept_name_keys = get_names_to_keys_dict(self.taxonomy_embedding_store.get_text_for_all_rows())
        self.entity_name_keys.update(self.concept_name_keys)

        self.ready_to_retrieve = True

    def get_query_embeddings(self, queries: List[str] | List[QuerySolution]):
        """
        Retrieves embeddings for given queries and updates the internal query-to-embedding mapping. The method determines whether each query
        is already present in the `self.query_to_embedding` dictionary under the keys 'triple' and 'passage'. If a query is not present in
        either, it is encoded into embeddings using the embedding model and stored.

        Args:
            queries List[str] | List[QuerySolution]: A list of query strings or QuerySolution objects. Each query is checked for
            its presence in the query-to-embedding mappings.
        """
        all_query_strings = []
        for query in queries:
            if isinstance(query, QuerySolution) and (
                    query.question not in self.query_to_embedding['triple'] or query.question not in
                    self.query_to_embedding['paragraph'] or query.question not in self.query_to_embedding['concept']):
                all_query_strings.append(query.question)
            elif query not in self.query_to_embedding['triple'] or query not in self.query_to_embedding['paragraph'] or query not in self.query_to_embedding['concept']:
                all_query_strings.append(query)

        if len(all_query_strings) > 0:
            # get all query embeddings
            logger.info(f"Encoding {len(all_query_strings)} queries for query_to_fact.")
            query_embeddings_for_triple = self.embedding_model.batch_encode(all_query_strings,
                                                                            instruction=get_query_instruction('query_to_fact'))
            for query, embedding in zip(all_query_strings, query_embeddings_for_triple):
                self.query_to_embedding['triple'][query] = embedding

            logger.info(f"Encoding {len(all_query_strings)} queries for query_to_passage.")
            query_embeddings_for_passage = self.embedding_model.batch_encode(all_query_strings,
                                                                             instruction=get_query_instruction('query_to_passage'))
            for query, embedding in zip(all_query_strings, query_embeddings_for_passage):
                self.query_to_embedding['paragraph'][query] = embedding

            # logger.info(f"Encoding {len(all_query_strings)} queries for query_to_concept.")
            # query_embeddings_for_concept = self.embedding_model.batch_encode(all_query_strings,
            #                                                                  instruction=get_query_instruction('query_to_concept'))
            # for query, embedding in zip(all_query_strings, query_embeddings_for_concept):
            #     self.query_to_embedding['concept'][query] = embedding

    def get_fact_scores(self, query: str) -> np.ndarray:
        """
        Retrieves and computes normalized similarity scores between the given query and pre-stored fact embeddings.

        Parameters:
        query : str
            The input query text for which similarity scores with fact embeddings
            need to be computed.

        Returns:
        numpy.ndarray
            A normalized array of similarity scores between the query and fact
            embeddings. The shape of the array is determined by the number of
            facts.

        Raises:
        KeyError
            If no embedding is found for the provided query in the stored query
            embeddings dictionary.
        """
        logger.info("[GRAPH] Extracting fact scores")
        query_embedding = self.query_to_embedding['triple'].get(query, None)
        if query_embedding is None:
            query_embedding = self.embedding_model.batch_encode(query, instruction=get_query_instruction('query_to_fact'))

        # Check if there are any facts
        if len(self.fact_embeddings) == 0:
            logger.warning("No facts available for scoring. Returning empty array.")
            return np.array([])
            
        try:
            query_fact_scores = np.dot(self.fact_embeddings, query_embedding.T) # shape: (#facts, )
            query_fact_scores = np.squeeze(query_fact_scores) if query_fact_scores.ndim == 2 else query_fact_scores
            query_fact_scores = min_max_normalize(query_fact_scores)
            return query_fact_scores
        except Exception as e:
            logger.error(f"Error computing fact scores: {str(e)}")
            return np.array([])

    def dense_passage_retrieval(self, query: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Conduct dense passage retrieval to find relevant documents for a query.

        This function processes a given query using a pre-trained embedding model
        to generate query embeddings. The similarity scores between the query
        embedding and passage embeddings are computed using dot product, followed
        by score normalization. Finally, the function ranks the documents based
        on their similarity scores and returns the ranked document identifiers
        and their scores.

        Parameters
        ----------
        query : str
            The input query for which relevant passages should be retrieved.

        Returns
        -------
        tuple : Tuple[np.ndarray, np.ndarray]
            A tuple containing two elements:
            - A list of sorted document identifiers based on their relevance scores.
            - A numpy array of the normalized similarity scores for the corresponding
              documents.
        """
        print("[INFO] Calculating DPR scores ...")
        # Encode query
        query_embedding = self.query_to_embedding['paragraph'].get(query, None)
        if query_embedding is None:
            query_embedding = self.embedding_model.batch_encode(query, instruction=get_query_instruction('query_to_passage'))
            
        # Compute similarity scores for passages
        query_doc_scores = np.dot(self.paragraph_embeddings, query_embedding.T)
        query_doc_scores = np.squeeze(query_doc_scores) if query_doc_scores.ndim == 2 else query_doc_scores
        query_doc_scores = min_max_normalize(query_doc_scores)
        # logger.info(f"PASSAGES: {len(self.passage_embeddings)} --- QUERY DOC: {len(query_doc_scores)}")

        # Adjust scores and rank results
        sorted_doc_ids = np.argsort(query_doc_scores)[::-1]
        sorted_doc_scores = query_doc_scores[sorted_doc_ids.tolist()]
        
        return sorted_doc_ids, sorted_doc_scores
    
    def get_top_k_weights(self,
                          link_top_k: int,
                          all_phrase_weights: np.ndarray,
                          linking_score_map: Dict[str, float]) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        This function filters the all_phrase_weights to retain only the weights for the
        top-ranked phrases in terms of the linking_score_map. It also filters linking scores
        to retain only the top `link_top_k` ranked nodes. Non-selected phrases in phrase
        weights are reset to a weight of 0.0.

        Args:
            link_top_k (int): Number of top-ranked nodes to retain in the linking score map.
            all_phrase_weights (np.ndarray): An array representing the phrase weights, indexed
                by phrase ID.
            linking_score_map (Dict[str, float]): A mapping of phrase content to its linking
                score, sorted in descending order of scores.

        Returns:
            Tuple[np.ndarray, Dict[str, float]]: A tuple containing the filtered array
            of all_phrase_weights with unselected weights set to 0.0, and the filtered
            linking_score_map containing only the top `link_top_k` phrases.
        """
        # choose top ranked nodes in linking_score_map
        linking_score_map = dict(sorted(linking_score_map.items(), key=lambda x: x[1], reverse=True)[:link_top_k])

        # only keep the top_k phrases in all_phrase_weights
        top_k_phrases = set(linking_score_map.keys())
        top_k_phrases_keys = []
        for top_k_phrase in top_k_phrases:
            top_k_phrases_keys.append(self.entity_name_keys[top_k_phrase])
        # top_k_phrases_keys = set(
        #     [compute_mdhash_id(content=top_k_phrase, prefix="entity_") for top_k_phrase in top_k_phrases])

        for phrase_key in self.node_name_to_vertex_idx:
            if phrase_key not in top_k_phrases_keys:
                phrase_id = self.node_name_to_vertex_idx.get(phrase_key, None)
                if phrase_id is not None:
                    all_phrase_weights[phrase_id] = 0.0

        assert np.count_nonzero(all_phrase_weights) == len(linking_score_map.keys())
        return all_phrase_weights, linking_score_map

    def graph_search_with_fact_entities(self, query: str,
                                        link_top_k: int,
                                        query_fact_scores: np.ndarray,
                                        top_k_facts: List[Tuple],
                                        top_k_fact_indices: List[str],
                                        passage_node_weight: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes document scores based on fact-based similarity and relevance using personalized
        PageRank (PPR) and dense retrieval models. This function combines the signal from the relevant
        facts identified with passage similarity and graph-based search for enhanced result ranking.

        Parameters:
            query (str): The input query string for which similarity and relevance computations
                need to be performed.
            link_top_k (int): The number of top phrases to include from the linking score map for
                downstream processing.
            query_fact_scores (np.ndarray): An array of scores representing fact-query similarity
                for each of the provided facts.
            top_k_facts (List[Tuple]): A list of top-ranked facts, where each fact is represented
                as a tuple of its subject, predicate, and object.
            top_k_fact_indices (List[str]): Corresponding indices or identifiers for the top-ranked
                facts in the query_fact_scores array.
            passage_node_weight (float): Default weight to scale passage scores in the graph.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing two arrays:
                - The first array corresponds to document IDs sorted based on their scores.
                - The second array consists of the PPR scores associated with the sorted document IDs.
        """
        print("[GRAPH] Performing graph search with facts")
        #Assigning phrase weights based on selected facts from previous steps.
        linking_score_map = {}  # from phrase to the average scores of the facts that contain the phrase
        phrase_scores = {}  # store all fact scores for each phrase regardless of whether they exist in the knowledge graph or not
        phrase_weights = np.zeros(len(list(self.graph.nodes()))) # np.zeros(len(self.graph.vs['name']))
        passage_weights = np.zeros(len(list(self.graph.nodes()))) # np.zeros(len(self.graph.vs['name']))

        for rank, f in enumerate(top_k_facts):
            subject_phrase = f[0].lower()
            predicate_phrase = f[1].lower()
            object_phrase = f[2].lower()
            fact_score = query_fact_scores[
                top_k_fact_indices[rank]] if query_fact_scores.ndim > 0 else query_fact_scores
            for phrase in [subject_phrase, object_phrase]:
                # phrase_key = compute_mdhash_id(
                #     content=phrase,
                #     prefix="entity_"
                # )
                phrase_key = self.entity_name_keys[phrase]
                phrase_id = self.node_name_to_vertex_idx.get(phrase_key, None)

                if phrase_id is not None:
                    phrase_weights[phrase_id] = fact_score

                    if self.ent_node_to_num_chunk[phrase_key] != 0:
                        phrase_weights[phrase_id] /= self.ent_node_to_num_chunk[phrase_key]

                if phrase not in phrase_scores:
                    phrase_scores[phrase] = []
                phrase_scores[phrase].append(fact_score)

        # calculate average fact score for each phrase
        for phrase, scores in phrase_scores.items():
            linking_score_map[phrase] = float(np.mean(scores))

        if link_top_k:
            phrase_weights, linking_score_map = self.get_top_k_weights(link_top_k,
                                                                        phrase_weights,
                                                                        linking_score_map)  # at this stage, the length of linking_scope_map is determined by link_top_k

        #Get passage scores according to chosen dense retrieval model
        dpr_sorted_doc_ids, dpr_sorted_doc_scores = self.dense_passage_retrieval(query)
        normalized_dpr_sorted_scores = min_max_normalize(dpr_sorted_doc_scores)

        for i, dpr_sorted_doc_id in enumerate(dpr_sorted_doc_ids.tolist()):
            passage_node_key = self.paragraph_node_keys[dpr_sorted_doc_id]
            passage_dpr_score = normalized_dpr_sorted_scores[i]
            passage_node_id = self.node_name_to_vertex_idx[passage_node_key]
            passage_weights[passage_node_id] = passage_dpr_score * passage_node_weight
            passage_node_text = self.paragraph_embedding_store.get_row(passage_node_key)["content"]
            linking_score_map[passage_node_text] = passage_dpr_score * passage_node_weight

        #Combining phrase and passage scores into one array for PPR
        node_weights = phrase_weights + passage_weights

        #Recording top 30 facts in linking_score_map
        if len(linking_score_map) > 30:
            linking_score_map = dict(sorted(linking_score_map.items(), key=lambda x: x[1], reverse=True)[:30])

        assert sum(node_weights) > 0, f'No phrases found in the graph for the given facts: {top_k_facts}'

        #Running PPR algorithm based on the passage and phrase weights previously assigned
        ppr_sorted_doc_ids, ppr_sorted_doc_scores = self.run_ppr(node_weights, damping=self.damping_factor)

        assert len(ppr_sorted_doc_ids) == len(
            self.paragraph_node_idxs), f"Doc prob length {len(ppr_sorted_doc_ids)} != corpus length {len(self.paragraph_node_idxs)}"

        return ppr_sorted_doc_ids, ppr_sorted_doc_scores

    def rerank_facts(self, query: str, query_fact_scores: np.ndarray) -> Tuple[List[int], List[Tuple], dict]:
        """

        Returns:
            top_k_fact_indicies:
            top_k_facts:
            rerank_log (dict): {'facts_before_rerank': candidate_facts, 'facts_after_rerank': top_k_facts}
                - candidate_facts (list): list of link_top_k facts (each fact is a relation triple in tuple data type).
                - top_k_facts:


        """
        print("[GRAPH] Reranking facts")
        link_top_k: int = self.link_top_k

        candidate_fact_indices = np.argsort(query_fact_scores)[-link_top_k:][::-1].tolist()  # list of ranked link_top_k fact relative indices
        real_candidate_fact_ids = [self.fact_node_keys[idx] for idx in
                                   candidate_fact_indices]  # list of ranked link_top_k fact keys
        fact_row_dict = self.fact_embedding_store.get_rows(real_candidate_fact_ids)

        candidate_facts = [eval(re.sub(r"(?<=\w)'s|(?<=\w)s'(?![),])", '', fact_row_dict[id]['content'])) for id in real_candidate_fact_ids]  # list of link_top_k facts (each fact is a relation triple in tuple data type)
        # candidate_facts = [eval(fact_row_dict[id]['content']) for id in real_candidate_fact_ids] 

        top_k_fact_indices, top_k_facts, reranker_dict = self.rerank_filter(query,
                                                                             candidate_facts,
                                                                             candidate_fact_indices,
                                                                             len_after_rerank=link_top_k)

        rerank_log = {'facts_before_rerank': candidate_facts, 'facts_after_rerank': top_k_facts}

        return top_k_fact_indices, top_k_facts, rerank_log
    
    def run_ppr(self,
                reset_prob: np.ndarray,
                damping: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs Personalized PageRank (PPR) on a graph and computes relevance scores for
        nodes corresponding to document passages. The method utilizes a damping
        factor for teleportation during rank computation and can take a reset
        probability array to influence the starting state of the computation.

        Parameters:
            reset_prob (np.ndarray): A 1-dimensional array specifying the reset
                probability distribution for each node. The array must have a size
                equal to the number of nodes in the graph. NaNs or negative values
                within the array are replaced with zeros.
            damping (float): A scalar specifying the damping factor for the
                computation. Defaults to 0.5 if not provided or set to `None`.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing two numpy arrays. The
                first array represents the sorted node IDs of document passages based
                on their relevance scores in descending order. The second array
                contains the corresponding relevance scores of each document passage
                in the same order.
        """
        print("[GRAPH] Running PPR algorithm")
        if damping is None: damping = 0.5 # for potential compatibility
        
        reset_prob = np.where(np.isnan(reset_prob) | (reset_prob < 0), 0, reset_prob)
        personalization = {} # Create personalization dictionary from reset_prob
        for idx, node in enumerate(self.graph.nodes()):
            if reset_prob[idx] > 0:
                personalization[node] = reset_prob[idx]

        # Normalize personalization weights (NetworkX requires sum = 1)
        total = sum(personalization.values())
        if total > 0:
            personalization = {k: v/total for k, v in personalization.items()}
        else:
            # If all weights are zero, use uniform distribution
            personalization = None

        pagerank_scores_dict = nx.pagerank(
            self.graph,
            alpha=damping,  # damping factor
            personalization=personalization,  # reset probabilities
            weight='weight',  # edge weight attribute
            max_iter=100,  # maximum iterations
            tol=1e-06  # convergence tolerance
        )

        # Convert dict to list in same node order (to match igraph's list output)
        pagerank_scores = [pagerank_scores_dict.get(node, 0.0) for node in self.graph.nodes()]

        doc_scores = np.array([pagerank_scores[idx] for idx in self.paragraph_node_idxs])
        sorted_doc_ids = np.argsort(doc_scores)[::-1]
        sorted_doc_scores = doc_scores[sorted_doc_ids.tolist()]

        return sorted_doc_ids, sorted_doc_scores
