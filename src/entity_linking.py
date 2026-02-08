# This file is used to link extracted entities with taxonomy concepts

import os
import argparse
import datetime
import json
from collections import defaultdict
from tqdm import tqdm

from .utils.consts import *
from .utils.basic_utils import *
from .extract_nouns import Retriever

start_time = datetime.datetime.now()
print(f"Start time: {start_time}")

def run_entity_linking(report_name, llm, threshold):
    print("\n=== Experiment INFO ===")
    print("[INFO] Task: Entity Linking")
    print("[INFO] Report: ", report_name)

    input_dir = f"outputs/openie/{llm}"
    output_dir = f"outputs/postRAG/{llm}"
    os.makedirs(output_dir, exist_ok=True)
 
    RAG = Retriever(report=report_name)
    TAX = load_json_file(PATH["TAX"])
    
    preds = load_json_file(f"{input_dir}/{report_name}.json")
    post = defaultdict(list)
    
    print("Linking entities ...")
    for entity in tqdm(preds['entities']):
        uuid, score = RAG.retrieve_by_def(entity["canonical_name"], entity["definition"])
        if score > threshold:
            entity.update({"taxonomy_uuid": uuid, "score": score})

    with open(f"{output_dir}/{report_name}.json", "w") as f:
        json.dump(preds, f, indent=2)

    print(f"Time now: {datetime.datetime.now()}. Time elapsed: {datetime.datetime.now() - start_time}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', type=str, default='')
    parser.add_argument('--llm', type=str, default='Llama-3.3-70B-InstructB')
    parser.add_argument('--threshold', type=int, default=50)
    args = parser.parse_args()

    report_name = args.report
    threshold = args.threshold
    llm = args.llm

    run_entity_linking(report_name, llm, threshold)
    