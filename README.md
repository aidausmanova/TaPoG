# Taxonomy-Populated Graph for Evidence Retrieval in Domain-specific Text

### Abstract
Corporate sustainability reports (CSRs) are essential for accountability, enabling regulators, investors and NGOs to verify company claims and identify greenwashing. However, retrieving specific evidence from lengthy, jargon-dense texts is challenging. Standard embedding-based retrieval and RAG fail to capture nuances due to domain-specific terminology and cross-document references.

We present a taxonomy-populated graph-based retrieval method that integrates expert-curated domain knowledge into a unified knowledge graph. Domain taxonomy assists in filtering relevant entities and explicit taxonomy nodes ground graph entities for disambiguation.
Experiments on two sustainability reporting benchmarks, Climretrieve and SustainablQA, demonstrate 25.7\% and 32.1\% relative improvement in recall over baselines, confirming that explicit expert knowledge grounding enhances evidence retrieval for domain-specific queries.

---------
### Setup
1. Clone repository and setup environment
   ```bash
   conda create --name venv python=3.11
   conda activate venv
   pip install -r requirements.txt
   ```
3. The `data/reports/` folder contains sustainability reports. Original data can be downloaded from [ClimRetrieve](https://github.com/tobischimanski/ClimRetrieve/blob/main/Report-Level%20Dataset/ClimRetrieve_ReportLevel_V1.csv) and [SustainableQA](https://github.com/DataScienceUIBK/SustainableQA/tree/main/Data) repositories.
4. Create `outputs/` and `logs/` folders

### Running Experiments
There are four modules in the pipeline: noun extraction, triple extraction, entity linking and graph construction. Each module can be run separetly. To run the full pipeline with retrieval follow `run.sh` instructions.

1. Intitalize variables
   ```bash
   report=ReportName # name of the document from data/reports folder
   model=Llama-3.1-70B-Instruct # or any other llm
   ```
2. Running
   
   ```bash
   bash run.sh
   ```
