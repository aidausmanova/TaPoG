# This file is used to extract all named entities with LLMs

import os
import torch
import json
import uuid
import datetime
import argparse
from tqdm import tqdm
from collections import defaultdict

from .utils.consts import *
from .utils.basic_utils import *
from .extract_nouns import Retriever
from .llm.meta_llama import InfoExtractor

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"\n=== GPU {i} ===")
        props = torch.cuda.get_device_properties(i)
        print(f"Name: {props.name}")
        print(f"Total Memory: {props.total_memory / 1e9:.2f} GB")
else:
    print("CUDA not available")

start_time = datetime.datetime.now()
print(f"Time now: {start_time}")

def canonicalize_entity(entity_name, entity_label):
    """
        Function to disambiguate and canonicalize entities
    """

    if entity_label == 'organization': 
        normalized_entity_name = normalize_organization_name(entity_name.lower())
    else:
        # normalized = split_name_and_abbrev(entity_name.lower())
        normalized = basic_normalize(entity_name.lower())
        match = re.match(r"\(([^)]+)\)", entity_name)
        if not match:
            normalized = expand_abbreviations(normalized)
        normalized_entity_name = lemmatization(normalized)

    return normalized_entity_name

def global_entity_transform(input_data, canonicalize=False):
    """
    Function to build a deduplicated global entity map from extracted entities and relationships.
    """
    global_entities = {}  # canonical_name -> entity data
    entity_id_mapping = {}  # canonical_name -> entity uuid 
    global_relationships = []  # list of all relationships
    paragraph_index = defaultdict(list)  # paragraph_id -> list of entity names
    name_mapping = {}  # original_name -> canonical_name (per paragraph)
    
    for paragraph_id, content in input_data.items():
        entities = content.get('entities', [])
        local_mapping = {}
        for entity in entities:
            original_name = entity['name']
            if canonicalize: canonical_name = canonicalize_entity(original_name, entity['label'])
            else: canonical_name = original_name
            
            local_mapping[original_name] = canonical_name
            
            if canonical_name not in global_entities:
                entity_uuid = str(uuid.uuid4())
                entity_id_mapping[canonical_name] = entity_uuid
                
                global_entities[canonical_name] = {
                    'id': entity_uuid,
                    'canonical_name': canonical_name,
                    'label': entity['label'],
                    'definition': entity['description'],
                    'original_names': set([original_name]),
                    'mentioned_in_paragraphs': [paragraph_id],
                    'mention_count': 1
                }
            else:
                existing = global_entities[canonical_name]
                existing['original_names'].add(original_name)
                existing['mentioned_in_paragraphs'].append(paragraph_id)
                existing['mention_count'] += 1
                
                if entity['description'] not in existing['definition']:
                    existing['definition'] += f" | {entity['description']}"
            
            if canonical_name not in paragraph_index[paragraph_id]:
                paragraph_index[paragraph_id].append(canonical_name)
        
        name_mapping[paragraph_id] = local_mapping

    
    seen_relationships = set()
    for paragraph_id, content in input_data.items():
        relationships = content.get('relationships', [])
        local_mapping = name_mapping[paragraph_id]
        
        for rel in relationships:
            source_original = rel['source']
            target_original = rel['target']
            relation_type = rel['relation']
            
            source_canonical = local_mapping.get(source_original, source_original)
            source_id = entity_id_mapping.get(source_canonical)
            target_canonical = local_mapping.get(target_original, target_original)
            target_id = entity_id_mapping.get(target_canonical)
            
            if not source_id or not target_id:
                continue
            
            rel_signature = (source_id, relation_type, target_id)
            if rel_signature not in seen_relationships:
                global_relationships.append({
                    'source_id': source_id,
                    'source_name': source_canonical,
                    'relation': relation_type,
                    'target_id': target_id,
                    'target_name': target_canonical,
                    'source_paragraphs': [paragraph_id],
                    'frequency': 1
                })
                seen_relationships.add(rel_signature)
            else:
                for existing_rel in global_relationships:
                    if (existing_rel['source_id'] == source_id and 
                        existing_rel['relation'] == relation_type and 
                        existing_rel['target_id'] == target_id):
                        existing_rel['source_paragraphs'].append(paragraph_id)
                        existing_rel['frequency'] += 1
                        break
    
    new_global_entities = []
    for entity_data in global_entities.values():
        entity_data['original_names'] = list(entity_data['original_names'])
        new_global_entities.append(entity_data)
    
    print(f"Total paragraphs #: {len(input_data)}\nTotal entities #: {len(new_global_entities)}\nTotal relationships #: {len(global_relationships)}")
    return {
        'entities': new_global_entities,
        'entity_id_mapping': entity_id_mapping,
        'relationships': global_relationships,
        'paragraph_index': dict(paragraph_index)
    }

def run_open_ei(report_name, llm):
    print("\n=== Experiment INFO ===")
    print("[INFO] Task: Named Entity Extraction")
    print("[INFO] Report: ", report_name)
    print("[INFO] LLM model: ", llm)

    output_dir = f"outputs/openie/{llm}"
    conversation_dir = f"outputs/conversations/{llm}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(conversation_dir, exist_ok=True)
    print(f"Output dir: {output_dir}")

    MODEL = InfoExtractor(engine=llm, load_type='openai')
    # RETRIEVER = Retriever(report=report_name)

    with open(PATH["weakly_supervised"]['path']+report_name+"/corpus.json", "r") as f:
        data = json.load(f)

    ############# Run per paragraph batch #############
    text_chunks = []
    ids = []
    for d in data:
        text_chunks.append(d['text'])
        ids.append(d['idx'])

    retrieved_nodes = load_json_file(PATH["RAG"]["prev_retrieved"]+report_name+".json")
    filtered_retrieved_nodes =[]
    for node_k, node_v in retrieved_nodes.items():
        if len(node_v) > 0:
            filtered_retrieved_nodes.append(node_k)

    print("[INFO] Eaxtrating named entities ...")
    model_outputs, raw_outputs = MODEL.generate_responses(text_chunks, filtered_retrieved_nodes, 8)
    outputs = {idx: output for idx, output in zip(ids, model_outputs)}

    transformed_outputs = global_entity_transform(outputs)

    with open(f"{output_dir}/{report_name}.json", "w") as f:
        json.dump(transformed_outputs, f, indent=2)
    
    print(f"Time now: {datetime.datetime.now()}. Time elapsed: {datetime.datetime.now() - start_time}")


    ############# Run per each paragraph #############
    # text_chunks = []
    # for d in data:
    #     text_chunks.append((d['title']+" "+d['text'], d['idx']))
    # pbar = tqdm(text_chunks)
    # outputs = {}
    # conversations = {}
    # for text, idx in pbar:
    #     pbar.set_description(f"File [{report_name}] Paragraph [{idx}] Retrieving entities")
    #     retrieved_nodes = RETRIEVER.run(text)
    #     print(len(retrieved_nodes), " # nodes retreived")

    #     pbar.set_description(f"File [{report_name}] Chunk [{idx}] Running MODEL")
    #     output, conversation = MODEL.run(text, retrieved_nodes)
    #     outputs[idx] = output
    #     conversations[idx] = conversation
    #     pbar.set_description(f"Output paragraph [{idx}]: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', type=str, default='')
    parser.add_argument('--llm', type=str, default='Llama-3.3-70B-InstructB')
    args = parser.parse_args()

    report_name = args.report
    llm = args.llm

    run_open_ei(report_name, llm)
