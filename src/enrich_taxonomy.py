# This file is used to enrich taxonomy concept defintion with LLM 
# for NER in climate disclosure and based on its hierarchical relationships

import os
import json
import uuid
import torch
import transformers
import argparse
from tqdm import tqdm

from dotenv import load_dotenv
load_dotenv()


from prompts.prompt_template_manager import PROMPT_REFINE_DEFINITIONS, REFINE_DEFINITIONS_INSTRUCTIONS

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'   
os.environ['VLLM_TORCH_COMPILE_LEVEL'] = '0'     

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm', type=str, default='meta-llama/Llama-3.1-70B-Instruct')
    parser.add_argument('--taxonomy', type=str)
    parser.add_argument('--load_type', type=str, default='chatai')
    args = parser.parse_args()

    MODEL_NAME = args.llm
    LOAD_TYPE = args.load_type
    taxonomy_path = args.taxonomy

    if LOAD_TYPE == "chatai":
        # --------- OpenaAI Chat AI --------
        from openai import OpenAI

        client = OpenAI(
        api_key = os.environ['CHAT_AI_KEY'],
        base_url = os.environ['BASE_URL']
    )
    elif LOAD_TYPE == 'vllm':
        # --------- VLLM --------
        from vllm import LLM, SamplingParams

        sampling_params = SamplingParams(
            temperature=0.6,
            top_p=0.9,
            max_tokens=2048,
        )
        model = LLM(
            model=MODEL_NAME,
            task="generate",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.8,
            max_model_len=4096,
            enable_prefix_caching=True,
        )
    else:
        # --------- Huggingface --------
        model = transformers.pipeline(
                        "text-generation",
                        model=MODEL_NAME,
                        model_kwargs={"torch_dtype": torch.bfloat16},
                        device_map="auto",
                    )
        processor = [
            model.tokenizer.eos_token_id,
            model.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]
        model.tokenizer.pad_token_id = model.tokenizer.eos_token_id
    
    print(f"[INFO] Model {MODEL_NAME} loaded with {LOAD_TYPE}")
    print(f"[INFO] Loading taxonomy {taxonomy_path}")

    with open(taxonomy_path+".json", "r") as f:
        taxonomy_data = json.load(f)

    enriched_tax_data = {}
    entity_id = 0
    entity_uuid_dict = {}
    prompts = []
    conversations = []

    for tax_entity in taxonomy_data.values():
        entity_metadata = f"Label: {tax_entity['prefLabel']}\n Type: {tax_entity['ifrs_type']} with data type {tax_entity['data_type']}\nDefinition: {tax_entity['definition']}\nOntology path: {' > '.join(tax_entity['path_label'])}"
        prompt = PROMPT_REFINE_DEFINITIONS.format(metadata=entity_metadata)
        prompts.append(prompt)
        conversations.append({"role": "user", "content": prompt})

    print(f"Processing {len(prompts)} entities...")
    if LOAD_TYPE == 'chatai':
        # --------- OpenaAI Chat AI --------
        outputs = []
        for tax_entity in taxonomy_data.values():
            entity_metadata = f"Label: {tax_entity['prefLabel']}\n Type: {tax_entity['ifrs_type']} with data type {tax_entity['data_type']}\nDefinition: {tax_entity['definition']}\nOntology path: {' > '.join(tax_entity['path_label'])}"
            try:
                output = client.chat.completions.create(
                    messages=[{"role":"system","content":REFINE_DEFINITIONS_INSTRUCTIONS},
                            {"role":"user","content":f"Metadata: {entity_metadata}\nEdited Definition:"}],
                    model= "meta-llama-3.1-70b-instruct"
                )
                outputs.append(output.choices[0].message.content)
            except:
                outputs.append(None)
                continue
    elif LOAD_TYPE == 'vllm':
        # --------- VLLM --------
        outputs = model.generate(prompts, sampling_params=sampling_params)
    else:
        # --------- Huggingface --------
        outputs = model(
                        prompts,
                        max_new_tokens=4000,
                        pad_token_id=model.tokenizer.eos_token_id,
                        # eos_token_id=terminators,
                        do_sample=True,
                        temperature=0.6,
                        return_full_text=False,
                        batch_size=8,
                        # top_p=0.9,
                    )

    for tax_uuid, tax_entity, output in tqdm(zip(taxonomy_data.keys(), taxonomy_data.values(), outputs), total=len(taxonomy_data)):
        print(f"[INFO] Processing entity [{tax_entity['prefLabel']}]")
        print(f"[OUTPUT] {output}")
        if LOAD_TYPE == 'chatai':
            # --------- OpenaAI Chat AI --------
            pred_content = output
        if LOAD_TYPE == 'vllm':
            # --------- VLLM --------
            pred_content = output.outputs[0].text
        else:
            # --------- Huggingface --------
            try:
                pred_content = output["generated_text"]
            except:
                pred_content = output
        
        enriched_tax_data[tax_uuid] = {
            "id": entity_id,
            "ifrs_id": tax_entity['ifrs_id'],
            "prefLabel": tax_entity['prefLabel'],
            "type": tax_entity['type'],
            "substitutionGroup": tax_entity['substitutionGroup'],
            "is_abstract": tax_entity['abstract'],
            "uri": tax_entity['uri'],
            "ifrs_definition": tax_entity['definition'],
            "path_label": tax_entity['path_label'],
            "path_id": tax_entity['path_id'],
            "enriched_definition": pred_content
        }
        entity_id += 1
    
    model = MODEL_NAME.split("/")[1]
    with open(taxonomy_path+f"_enriched_{model}.json", "w") as f:
        json.dump(enriched_tax_data, f)

    print("[INFO] Enriched taxonomy saved")
