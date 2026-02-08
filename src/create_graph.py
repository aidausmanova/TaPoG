# This file is used to run graph construction task

import os
import argparse
from typing import List
import json
import datetime

from .graph.kg import ReportKnowledgeGraph
from .utils.consts import *
from .utils.basic_utils import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


start_time = datetime.datetime.now()
print(f"Start time: {start_time}")

def run_graph_build_retrieval(report_name, taxonomy, question_type=None):
    print("\n=== Experiment INFO ===")
    print("[INFO] Task: Graph Construction")
    print("[INFO] Report: ", report_name)
    
    graph = ReportKnowledgeGraph(report_name, taxonomy)
    
    print("[INFO] Starting retreival ...")
    all_samples = json.load(open(f"{PATH['weakly_supervised']['path']}{report_name}/gold.json", "r"))
    samples = []
    if question_type:
        for s in all_samples:
            if s['type'] == question_type:
                samples.append(s)
    else:
        samples = all_samples
    
    all_queries = [s['question'] for s in samples]

    gold_docs = get_gold_docs(samples, report_name)
    gold_answers = get_gold_answers(samples)

    assert len(all_queries) == len(gold_docs) == len(gold_answers), "Length of queries, gold_docs, and gold_answers should be the same."

    if gold_docs is not None:
        queries, overall_retrieval_result = graph.retrieve(queries=all_queries, num_to_retrieve=10, gold_docs=gold_docs)

    else:
        queries = graph.retrieve(queries=all_queries, num_to_retrieve=15)

    print(f"Time now: {datetime.datetime.now()}. Time elapsed: {datetime.datetime.now() - start_time}")

    return overall_retrieval_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ClimateAGE Graph")
    parser.add_argument('--report', type=str, default='')
    parser.add_argument('--taxonomy', type=str)
    parser.add_argument('--question_type', type=str)
    args = parser.parse_args()
    report_name = args.report
    taxonomy = args.taxonomy
    question_type = args.question_type

    run_graph_build_retrieval(report_name, taxonomy, question_type)
    