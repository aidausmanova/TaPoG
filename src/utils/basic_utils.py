import json
import re
import spacy
import logging
import unicodedata
import numpy as np
from typing import Dict, List, Set, Tuple, Optional, Callable
from sklearn.metrics import f1_score

from src.utils.consts import ABBREVIATIONS, ORG_SUFFIXES

nlp = spacy.load("en_core_web_sm")


text_template = "<heading>{}</heading>\n{}\n"

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with a specific name and optional file logging.

    Args:
        name (str): Logger name, typically the module's `__name__`.
        log_file (str): Log file name. If None, defaults to "<name>.log" under the logs directory.
        level (int): Logging level (e.g., logging.DEBUG, logging.INFO).

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)

    return logger

def load_json_file(file_path):
    """
    Loads a JSON file and returns its contents as a Python dictionary.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        dict: The contents of the JSON file as a dictionary.
    """
    with open(file_path, "r") as file:
        data = json.load(file)

    return data

def get_gold_docs(samples: List, dataset_name: str = None) -> List:
    gold_docs = []
    for sample in samples:
        if 'supporting_facts' in sample:  # hotpotqa, 2wikimultihopqa
            gold_title = set([item[0] for item in sample['supporting_facts']])
            gold_title_and_content_list = [item for item in sample['context'] if item[0] in gold_title]
            if dataset_name.startswith('hotpotqa'):
                gold_doc = [item[0] + '\n' + ''.join(item[1]) for item in gold_title_and_content_list]
            else:
                gold_doc = [item[0] + '\n' + ' '.join(item[1]) for item in gold_title_and_content_list]
        elif 'contexts' in sample:
            gold_doc = [item['title'] + '\n' + item['text'] for item in sample['contexts'] if item['is_supporting']]
        else:
            assert 'paragraphs' in sample, "`paragraphs` should be in sample, or consider the setting not to evaluate retrieval"
            gold_paragraphs = []
            for item in sample['paragraphs']:
                if 'is_supporting' in item and item['is_supporting'] is False:
                    continue
                gold_paragraphs.append(item)
            # gold_doc = [{"idx": item['idx'], 'text': item['title'] + '\n' + (item['text'] if 'text' in item else item['paragraph_text'])} for item in gold_paragraphs]
            gold_doc = [item['idx'] for item in gold_paragraphs]

        gold_doc = list(set(gold_doc))
        gold_docs.append(gold_doc)
    return gold_docs

def get_gold_answers(samples):
    gold_answers = []
    for sample_idx in range(len(samples)):
        gold_ans = None
        sample = samples[sample_idx]

        if 'answer' in sample or 'gold_ans' in sample:
            gold_ans = sample['answer'] if 'answer' in sample else sample['gold_ans']
        elif 'reference' in sample:
            gold_ans = sample['reference']
        elif 'obj' in sample:
            gold_ans = set(
                [sample['obj']] + [sample['possible_answers']] + [sample['o_wiki_title']] + [sample['o_aliases']])
            gold_ans = list(gold_ans)
        assert gold_ans is not None
        if isinstance(gold_ans, str):
            gold_ans = [gold_ans]
        assert isinstance(gold_ans, list)
        gold_ans = set(gold_ans)
        if 'answer_aliases' in sample:
            gold_ans.update(sample['answer_aliases'])

        gold_answers.append(gold_ans)

    return gold_answers

def basic_normalize(text: str) -> str:
    if not text:
        return ""
    # Unicode normalization (NFKC - compatibility decomposition followed by canonical composition)
    text = unicodedata.normalize('NFKC', text)
    text = text.lower()
    text = ' '.join(text.split())
    text = re.sub(r'[^\w\s\-]', ' ', text)
    text = ' '.join(text.split())
    return text.strip()

def normalize_organization_name(name: str) -> str:
    normalized = basic_normalize(name)
    words = normalized.split()
    
    # Remove common suffixes
    filtered_words = []
    for word in words:
        if word not in ORG_SUFFIXES:
            filtered_words.append(word)
    
    # If all words were suffixes, keep original
    if not filtered_words:
        filtered_words = words
    return ' '.join(filtered_words)

def expand_abbreviations(text: str) -> str:
    words = text.split()
    expanded_words = []
    for word in words:
        # Check if word is an abbreviation
        clean_word = word.strip('.,;:!?')
        if clean_word in ABBREVIATIONS:
            expanded_words.append(ABBREVIATIONS[clean_word])
        else:
            expanded_words.append(word)
    return ' '.join(expanded_words)

def lemmatization(text: str) -> str:
    doc = nlp(text)
    lemmas = [token.lemma_ for token in doc]
    return ' '.join(lemmas)

def min_max_normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

def get_names_to_keys_dict(text):
    name_keys = {}
    for entity_key, entity_metadata in text.items():
        name = entity_metadata['content'].split('\n')[0].lower()
        name_keys[name] = entity_key
    return name_keys
