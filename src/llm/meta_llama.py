import os
import numpy as np
import torch
from tqdm import tqdm
import transformers
from vllm import SamplingParams, LLM
from string import Template
from transformers import PreTrainedTokenizer
from typing import (
    Optional,
    Union,
    List,
    TypedDict
)

from ..prompts.prompt_template_manager import *
from ..utils.basic_utils import *
from ..utils.consts import *

from dotenv import load_dotenv
load_dotenv()

logger = get_logger(__name__)


class TextChatMessage(TypedDict):
    """Representation of a single text-based chat message in the chat history."""
    role: str
    content: Union[str, Template]


def convert_text_chat_messages_to_input_ids(messages: List[TextChatMessage], tokenizer: PreTrainedTokenizer, add_assistant_header=True) -> List[List[int]]:
    prompt = tokenizer.apply_chat_template(
        conversation=messages,
        chat_template=None,
        tokenize=False,
        add_generation_prompt=True,
        continue_final_message=False,
        tools=None,
        documents=None,
    )
    encoded = tokenizer(prompt, add_special_tokens=False)
    return encoded['input_ids']


class InfoExtractor:
    def __init__(self, engine="Llama-3.3-70B-Instruct", n_shot=10, load_type='chatai', is_query_ie=False):
        """
        Initializes an instance of the class and its related components.

        Attributes
            model: LLM model.
            user_vllm (bool): Variable to set if model is loaded via VLLM. Alternatively, 
                model is loaded via transformers pipeline.

        Parameters
            exp (str): Name of the current experiment (e.g. 0_shot, no_rag, no_relation)
            engine (str): Name of the LLM model to be used for processing.
            n_shot (int): Number of few-shot examples.
            use_vllm (bool): Variable to set if model is loaded via VLLM. Alternatively, 
                model is loaded via transformers pipeline.
        """
        self.model = None
        self.load_type = load_type
        self.is_query_ie = is_query_ie

        # Load LLM model
        model_id = "meta-llama/" + engine
        print(f"Loading model from {model_id}")

        if self.load_type == 'vllm':
            # --------- VLLM --------
            self.sampling_params = SamplingParams(
                temperature=0.6,
                top_p=0.9,
                stop=[DELIMITERS["completion_delimiter"]],
                max_tokens=2048,
            )
            self.model = LLM(
                model=model_id,
                task="generate",
                tensor_parallel_size=1,
                gpu_memory_utilization=0.8,
                max_model_len=20000,
                enable_prefix_caching=True,
            )
        elif self.load_type == 'chatai':
            # --------- OpenaAI Chat AI --------
            from openai import OpenAI
            self.model = OpenAI(
                api_key = os.environ['CHAT_AI_KEY'],
                base_url = os.environ['BASE_URL']
            )
        else:
            # --------- Huggingface --------
            self.model = transformers.pipeline(
                "text-generation",
                model=model_id,
                torch_dtype=torch.bfloat16,
                device_map='auto',
            )
            self.processor = [
                self.model.tokenizer.eos_token_id,
                self.model.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            ]
            self.model.tokenizer.pad_token_id = self.model.tokenizer.eos_token_id

        # load n-shot examples
        examples = load_json_file(PATH["LLM"]["examples"])
        self.formatted_examples = ""
        for i, example in enumerate(examples[:n_shot]):
            if is_query_ie:
                example = re.sub(
                    r'##\n\("relationship"<\|>.*?\)\n', "", example, flags=re.DOTALL
                )
            self.formatted_examples += f"\nExample {i+1}:\n{example}"

    def canonicalize_entity(entity_name, entity_label):
        """
            Function to disambiguate and canonicalize entities
        """

        if entity_label == 'organization': 
            normalized_entity_name = normalize_organization_name(entity_name)
        else:
            normalized = basic_normalize(entity_name)
            normalized = expand_abbreviations(normalized)
            normalized_entity_name = lemmatization(normalized)

        return normalized_entity_name

    def parse_response(self, response, with_description=True):
        """
            Function to parse LLM response.
        """
        out = {"entities": [], "relationships": []}
        # trim the response to start from the first entity
        start_index = response.find('("')
        if start_index != 0:
            response = response[start_index:]

        # trim the response to end with <|COMPLETE|>
        # if not self.use_vllm:
        response = response.split(DELIMITERS["completion_delimiter"])[0]

        # split response into records
        response = response.split(DELIMITERS["record_delimiter"])
        response = [
            r.lstrip("\n").rstrip("\n").lstrip("(").rstrip(")") for r in response
        ]

        # split the response into items
        pattern = r"<\s*\|\s*>"
        response = [re.split(pattern, r) for r in response]
        for r in response:
            if "entity" in r[0]:
                if with_description:
                    if len(r) == 4:
                        out["entities"].append(
                            {"name": r[1], "label": r[2], "description": r[3]}
                        )
                elif len(r) == 3:
                    out["entities"].append({"name": r[1], "label": r[2]})

            elif "relationship" in r[0]:
                if len(r) == 4:
                    out["relationships"].append(
                        {"source": r[1], "target": r[2], "relation": r[3]}
                    )
        return out

    def run(self, text, retrieved_nodes):
        """
            Process one paragraph at a time.
        """
        potential_entities = list(retrieved_nodes.keys())
        potential_entities = ", ".join(potential_entities)

        prompt = self.PROMPT_TEMPLATE.format(
            **DELIMITERS,
            formatted_examples=self.formatted_examples,
            input_text=text.replace("{", "").replace("}", ""),
            potential_entities=potential_entities,
        ).format(**DELIMITERS)

        conversation = [{"role": "user", "content": prompt}]
        if self.model:
            if self.load_type == 'vllm':
                output = self.model.generate(
                    prompt, sampling_params=self.sampling_params
                )
                pred_content = output[0].outputs[0].text
            else:
                outputs = self.model(
                    prompt,
                    max_new_tokens=4000,
                    pad_token_id=self.model.tokenizer.eos_token_id,
                    do_sample=True,
                    return_full_text=False,
                    temperature=0.6,
                    batch_size=8,
                )
                pred_content = outputs[0]["generated_text"]

        conversation.append({"role": "assistant", "content": pred_content})
        response = self.parse_response(pred_content)
        return response, conversation

    def run_batch(self, texts, retrieved_nodes_list):
        """
            Process multiple paragraphs at once.
        """
        print(f"[INFO] Processing {len(texts)} paragraphs")
       
        if self.load_type == 'chatai':
            # --------- OpenaAI Chat AI --------
            outputs = []
            for text, retrieved_nodes in zip(texts, retrieved_nodes_list):
                potential_entities = ", ".join(retrieved_nodes.keys())
                entity_metadata = f"Label: {tax_entity['prefLabel']}\n Type: {tax_entity['ifrs_type']} with data type {tax_entity['data_type']}\nDefinition: {tax_entity['definition']}\nOntology path: {' > '.join(tax_entity['path_label'])}"
                try:
                    output = self.model.chat.completions.create(
                        messages=[{"role":"system","content":REFINE_DEFINITIONS_INSTRUCTIONS},
                                {"role":"user","content":f"Metadata: {entity_metadata}\nEdited Definition:"}],
                        model= "meta-llama-3.1-70b-instruct"
                    )
                    outputs.append(output.choices[0].message.content)
                except:
                    outputs.append(None)
                    continue
        else:
            # --------- Huggingface --------
            prompts = []
            for text, retrieved_nodes in zip(texts, retrieved_nodes_list):
                potential_entities = ", ".join(retrieved_nodes.keys())
                prompt = self.PROMPT_TEMPLATE.format(
                    **DELIMITERS,
                    formatted_examples=self.formatted_examples,
                    input_text=text.replace("{", "").replace("}", ""),
                    potential_entities=potential_entities,
                ).format(**DELIMITERS)
                prompts.append(prompt)
            
            outputs = self.model(
                prompts,
                max_new_tokens=4000,
                pad_token_id=self.model.tokenizer.eos_token_id,
                do_sample=True,
                return_full_text=False,
                temperature=0.6,
            )
            
            # conversations = [{"input": prompt, "output": out["generated_text"]} for prompt, out in zip(prompts, outputs)]
            # responses = [self.parse_response(out["generated_text"]) for out in outputs]
            responses = []
            conversations = []
            for prompt, output in zip(prompts, outputs):
                pred_content = output["generated_text"]
                response = self.parse_response(pred_content)
                conversation = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": pred_content}
                ]
                responses.append(response)
                conversations.append(conversation)
            return responses, conversations

    def process(self, example):
        user_message = {"role": "user", "content": example}
        messages = [user_message]
        return self.model.tokenizer.apply_chat_template(
            messages, tokenize=False
        )

    def generate_responses(self, texts, retrieved_nodes_list=None, batch_size=8):
        """
            Function to extract named entities from the text.
        """
        print("[INFO] Extracting triples ...")
        results, raw_output = [], []
        if not retrieved_nodes_list:
            retrieved_nodes_list = [""] * len(texts) # If no potential entities provided

        if self.load_type == 'chatai':
            # --------- OpenaAI Chat AI --------
            for text, retrieved_nodes in tqdm(zip(texts, retrieved_nodes_list)):
                prompt = PROMPT_TEMPLATE_INSTRUCTIONS.format(
                    **DELIMITERS,
                    formatted_examples=self.formatted_examples
                )
                try:
                    input_text = text.replace("{", "").replace("}", "")
                    if self.is_query_ie:
                        messages = [{"role":"system","content":prompt},
                                {"role":"user","content":f"Text: {input_text}\n######################\nOutput:"}]
                    else:
                        potential_entities = ", ".join(retrieved_nodes)
                        messages = [{"role":"system","content":prompt},
                                {"role":"user","content":f"Text: {input_text}\nPotential Entities: {potential_entities}\n######################\nOutput:"}]
                        
                    output = self.model.chat.completions.create(messages=messages, model= "meta-llama-3.1-70b-instruct")
                    results.append(self.parse_response(output.choices[0].message.content))
                    raw_output.append(output.choices[0].message.content)
                except:
                    results.append(None)
                    raw_output.append(None)
                    continue
        else:
            prompts = []
            for text, retrieved_nodes in tqdm(zip(texts, retrieved_nodes_list)):
                # potential_entities = ", ".join(retrieved_nodes.keys())
                potential_entities = ", ".join(retrieved_nodes)
                prompt = self.PROMPT_TEMPLATE.format(
                    **DELIMITERS,
                    formatted_examples=self.formatted_examples,
                    input_text=text.replace("{", "").replace("}", ""),
                    potential_entities=potential_entities,
                ).format(**DELIMITERS)
                prompts.append(prompt)
            dataset = [self.process(example) for example in prompts]
            
            
            for n_batch in tqdm(range(len(dataset) // batch_size + 1)):
                batch = dataset[batch_size * n_batch : batch_size * (n_batch + 1)]
                if len(batch) == 0:
                    continue
                responses = self.model(
                    batch,
                    batch_size=batch_size,
                    max_new_tokens=4000,
                    do_sample=False,
                    num_beams=1,
                    return_full_text=False,
                )
                for response in responses:
                    results.append(self.parse_response(response[0]["generated_text"]))
                    raw_output.append(response[0]["generated_text"])
        return results, raw_output
    
    def infer(self, messages: List[TextChatMessage], max_tokens=2048):
        messages_list = [messages]
        if self.load_type == 'chatai':
            # --------- OpenaAI Chat AI --------
            logger.info(f"Calling OpenAI Chat AI, # of messages {len(messages)}")
            generate_params = {
                "model": "meta-llama-3.1-70b-instruct",
                "max_completion_tokens":  400,
                "n": 1,
                "seed": 0,
                "temperature": 0.0,
                "messages": messages,
                "max_tokens": 400
            }
            try:
                output = self.model.chat.completions.create(**generate_params)
                response = output.choices[0].message.content
                metadata = {
                    "prompt_tokens": output.usage.prompt_tokens, 
                    "completion_tokens": output.usage.completion_tokens,
                    "finish_reason": output.choices[0].finish_reason,
                }
            except:
                response = None
                metadata = None
        elif self.load_type == 'vllm':
            # --------- VLLM --------
            prompt_ids = convert_text_chat_messages_to_input_ids(messages_list, self.tokenizer)
            vllm_output = self.model.generate(prompt_token_ids=prompt_ids,  sampling_params=SamplingParams(max_tokens=max_tokens, temperature=0))
            response = vllm_output[0].outputs[0].text
            prompt_tokens = len(vllm_output[0].prompt_token_ids)
            completion_tokens = len(vllm_output[0].outputs[0].token_ids )
            metadata = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens
            }

        return response
