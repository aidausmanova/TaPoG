# This file contains prompts used in the pipeline

REFINE_DEFINITIONS_INSTRUCTIONS = """Given the following metadata about an entity in a climate disclosure ontology, which may include the entity’s name, entity's data type, ontology path,
and a definition (which may be missing), please develop an edited definition suitable for a named entity recognition (NER)
task in climate disclosure. The definition should be concise, clear, and limited to 150 tokens. Ensure it is precise and
emphasizes the entity’s unique aspects, avoiding overly general descriptions that could apply to multiple entities. Do not explain;
only provide the edited definition."""

PROMPT_REFINE_DEFINITIONS = """Given the following metadata about an entity in a climate disclosure ontology, which may include the entity’s name, entity's data type, ontology path,
and a definition (which may be missing), please develop an edited definition suitable for a named entity recognition (NER)
task in climate disclosure. The definition should be concise, clear, and limited to 150 tokens. Ensure it is precise and
emphasizes the entity’s unique aspects, avoiding overly general descriptions that could apply to multiple entities. Do not explain;
only provide the edited definition.
Metadata: {metadata}
Edited Definition:"""

# =========== base ===========
PROMPT_TEMPLATE = """{section_delimiter}Goal{section_delimiter}
Given a text document with a preliminary list of potential entities, verify, and identify all entities of the specified types within the text. Note that the initial list may contain missing or incorrect entities. 

{section_delimiter}Entity Types and Definitions{section_delimiter}
An entity is legal or reporting organization, an academic or political institution or a commercial company.
A report is formal document or filing that communicates an organization’s information, performance metrics, and narrative disclosures to stakeholders.
A plan is defined set of actions, milestones, and timelines describing how an entity will achieve its goals.
A resource refers to an input that supports an entity’s operations.
A disclosure topic is distinct subject area or theme of disclosure that captures specific issure.
An event is a discrete occurrence with environmental, financial, or operational implications related to climate.
An impact is an effect that an entity’s activities, products, or services have on the natural environment.
A location is a place on Earth, a location within Earth, a vertical location, or a location outside of the Earth.
A methodology is structured approach, model, or set of procedures an organization uses to measure, calculate, estimate, or assess its risks and impacts.

{section_delimiter}Steps{section_delimiter}
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity
- entity_type: One of the following types: [project, location, model, experiment, platform, instrument, provider, variable]
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

2. Return output in English as a single list of all the entities identified in steps 1. Use **{record_delimiter}** as the list delimiter. Do not output any code or steps for solving the question.

4. When finished, output {completion_delimiter}

######################
{section_delimiter}Examples{section_delimiter}
{formatted_examples}
######################
{section_delimiter}Real Data{section_delimiter}
######################
Text: {input_text}
Potential Entities: {potential_entities}
######################
Output:
"""

PROMPT_TEMPLATE_INSTRUCTIONS = """{section_delimiter}Goal{section_delimiter}
Given a text document with a preliminary list of potential entities, verify, and identify all entities of the specified types within the text. Note that the initial list may contain missing or incorrect entities. 

{section_delimiter}Entity Types and Definitions{section_delimiter}
An entity is legal or reporting organization, an academic or political institution or a commercial company.
A report is formal document or filing that communicates an organization’s information, performance metrics, and narrative disclosures to stakeholders.
A plan is defined set of actions, milestones, and timelines describing how an entity will achieve its goals.
A resource refers to an input that supports an entity’s operations.
A disclosure topic is distinct subject area or theme of disclosure that captures specific issure.
An event is a discrete occurrence with environmental, financial, or operational implications related to climate.
An impact is an effect that an entity’s activities, products, or services have on the natural environment.
A location is a place on Earth, a location within Earth, a vertical location, or a location outside of the Earth.
A methodology is structured approach, model, or set of procedures an organization uses to measure, calculate, estimate, or assess its risks and impacts.

{section_delimiter}Steps{section_delimiter}
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity
- entity_type: One of the following types: [project, location, model, experiment, platform, instrument, provider, variable]
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

2. Return output in English as a single list of all the entities identified in steps 1. Use **{record_delimiter}** as the list delimiter. Do not output any code or steps for solving the question.

4. When finished, output {completion_delimiter}

######################
{section_delimiter}Examples{section_delimiter}
{formatted_examples}
"""

def get_query_instruction(linking_method):
    instructions = {
        'ner_to_node': 'Given a phrase, retrieve synonymous or relevant phrases that best match this phrase.',
        'query_to_node': 'Given a question, retrieve relevant phrases that are mentioned in this question.',
        'query_to_fact': 'Given a question, retrieve relevant triplet facts that matches this question.',
        'query_to_sentence': 'Given a question, retrieve relevant sentences that best answer the question.',
        'query_to_passage': 'Given a question, retrieve relevant documents that best answer the question.',
        'query_to_concept': 'Given a question, retrieve relevant concept that best aligns with the question.',
    }
    default_instruction = 'Given a question, retrieve relevant documents that best answer the question.'
    return instructions.get(linking_method, default_instruction)
    