#!/bin/bash
#SBATCH --gpus-per-node=2
#SBATCH --constraint=48GB
#SBATCH --job-name=climateage
#SBATCH -o logs/run_%j.out # STDOUT

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

report='Netflix ESG Report 2022'
model='Llama-3.1-70B-Instruct'

python -m src.extract_nouns --report $report
python -m src.open_ie --report $report --llm $model
python -m src.entity_linking --report $report --llm $model --threshold 50
python -m src.create_graph --report $report --taxonomy 'ifrs_sds_taxonomy' #--question_type 'Descriptive, non-factoid' #'Factoid'
