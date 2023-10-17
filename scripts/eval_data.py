import os
import random
import json
import csv
import sys
import os
import logging
import xml.etree.ElementTree as ET
from collections import defaultdict
import re
import numpy as np

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def load_jsonl(path):
    data = []
    with open(path) as f:
        # bp()
        for line in f:
            data.append(json.loads(line))
    return data


def load_test_data(args):
    examples, closed_label_space = get_examples(args.dataset_name, args.split, args.dataset_dir)
    random.seed(0)
    if args.n_sample:
        if args.n_sample > len(examples):
            args.n_sample = len(examples)
        random.shuffle(examples)
        index_value = random.sample(list(range(0, len(examples))), args.n_sample)
        examples_sample = []
        for i in index_value:
            examples_sample.append(examples[i])
        examples = examples_sample
    return examples, closed_label_space


def get_examples(dataset_name, split, stem):
    if dataset_name == 'rte':
        examples = load_examples_rte(f'{stem}/{split}.jsonl')
        closed_label_space = True
    elif dataset_name == 'sst2':
        examples = load_examples_sst2(f'{stem}/{split}.tsv')
        closed_label_space = True
    elif dataset_name == 'amazon':
        examples = load_examples_amazon(f'{stem}/{split}.tsv')
        closed_label_space = True
    elif dataset_name == 'agn':
        examples = load_examples_agn(f'{stem}/{split}.csv')
        closed_label_space = True
    elif dataset_name == 'dbpedia':
        examples = load_examples_dbpedia(f'{stem}/{split}.csv')
        closed_label_space = True
    elif dataset_name == 'yelp':
        examples = load_examples_yelp(f'{stem}/{split}.jsonl')
        closed_label_space = True
    elif dataset_name == 'rotten_tomatoes':
        examples = load_examples_rotten_tomatoes(f'{stem}/{split}.jsonl')
        closed_label_space = True
    elif dataset_name == 'hyp':
        examples = load_examples_hyp(f'{stem}/{split}.csv')
        closed_label_space = True
    elif dataset_name == 'cr':
        examples = load_examples_cr(f'{stem}/{split}.csv')
        closed_label_space = True
    elif dataset_name == 'mr':
        examples = load_examples_mr(f'{stem}/{split}.csv')
        closed_label_space = True
    elif dataset_name == 'amazon':
        examples = load_examples_amazon(f'{stem}/{split}.csv')
        closed_label_space = True
    else:
        raise ValueError(f'Unknown dataset {dataset_name}')
    return examples, closed_label_space


def load_label(path):
    label2token = defaultdict(list)
    with open(path) as f:
        for i, line in enumerate(f):
            for w in line.strip().split(", "):
                label2token[i].append(f" {w}")

    # trim to the same length
    min_len = min([len(v) for v in label2token.values()])
    for k, v in label2token.copy().items():
        label2token[k] = v[:min_len]
    # print("label2token: ", label2token)
    return label2token

def load_examples_rotten_tomatoes(path):
    label2index = [' negative', ' positive']
    label_list = [' terrible', ' great']
    label_path = "data_eval/fuzzy_verbalizer/sentiment_verbalizer.txt"
    label2synonym = load_label(label_path)
    prompt = " It was"

    '''
    load train examples
    '''
    icl_str = ""
    examples = []
    with open(path, 'r') as json_file:
        json_list = list(json_file)

        for row in json_list:
            row = json.loads(row)
            label_str = " " + row["output"]
            label = label2index.index(label_str)
            summary = row["input"]
            premise = f'{summary}{prompt}'
            options = []
            for h in label_list:
                o = {}
                o['premise'] = icl_str + premise
                o['knn_premise'] = premise
                o['hypothesis'] = h.lower()
                o['uncond_premise'] = prompt
                o['uncond_hypothesis'] = h.lower()
                options.append(o)
            similarity = float(row["similarity"]) if "similarity" in row else None
            examples.append({'options': options, 'label': label, "sim": similarity, 'label2synonym': label2synonym, 'label_list': label_list})
    return examples

def load_examples_yelp(path):
    label2index = [' negative', ' positive']
    label_list = [' terrible', ' great']
    label_path = "data_eval/fuzzy_verbalizer/sentiment_verbalizer.txt"
    label2synonym = load_label(label_path)
    prompt = " It was"
    icl_str = ""
    
    examples = []
    with open(path, 'r') as json_file:
        json_list = list(json_file)

        for row in json_list:
            row = json.loads(row)
            label_str = " " + row["output"]
            label = label2index.index(label_str)
            summary = row["input"]
            premise = f'{summary}{prompt}'
            options = []
            for h in label_list:
                o = {}
                o['premise'] = icl_str + premise
                o['knn_premise'] = premise
                o['hypothesis'] = h.lower()
                o['uncond_premise'] = prompt
                o['uncond_hypothesis'] = h.lower()
                options.append(o)
            similarity = float(row["similarity"]) if "similarity" in row else None
            examples.append({'options': options, 'label': label, "sim": similarity, 'label2synonym': label2synonym, 'label_list': label_list})
    return examples

def load_examples_dbpedia(path):
    hypotheses = (
        " company",
        " school",
        " artist",
        " athlete",
        " politics",
        " transportation",
        " building",
        " river",
        " village",
        " animal",
        " plant",
        " album",
        " film",
        " book"
    )
    label_path = "data_eval/fuzzy_verbalizer/dbpedia_verbalizer.txt"
    label2synonym = load_label(label_path)
    prompt =" The topic of a text is"
    examples = []
    with open(path) as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            d = {}
            label = int(row['Class'])-1
            premise = f"{row['Text']}{prompt}"
            options = []
            for h in hypotheses:
                o = {}
                o['premise'] = premise
                o['knn_premise'] = premise
                o['hypothesis'] = h
                o['uncond_premise'] = prompt
                o['uncond_hypothesis'] = h
                options.append(o)
            examples.append({'options' : options, 'label' : label, 'label2synonym': label2synonym, 'label_list': hypotheses})
    return examples

def load_examples_rte(path):
    label_path = "data_eval/fuzzy_verbalizer/rte_verbalizer.txt"
    label2synonym = load_label(label_path)
    label_list = [' true', ' false']
    prompt = ' true or false?\n answer:'

    icl_str = ""

    data = []
    with open(path) as f:
        for line in f:
            data += [json.loads(line)]

    examples = []
    for d in data:
        premise = f" {d['premise']}\n question: {d['hypothesis']} {prompt}"
        options = []
        for h in label_list:
            o = {}
            o['premise'] = icl_str + premise
            o['knn_premise'] = premise
            o['hypothesis'] = h
            o['uncond_premise'] = prompt
            o['uncond_hypothesis'] = h
            options.append(o)
        label = 0 if d['label'] == 'entailment' else 1
        examples.append({'options' : options, 'label' : label, 'label2synonym': label2synonym, 'label_list': label_list})
    return examples



def load_examples_amazon(path):
    label_list = [' terrible', ' great']
    label_path = "data_eval/fuzzy_verbalizer/sentiment_verbalizer.txt"
    label2synonym = load_label(label_path)
    prompt = " It was"

    data = load_jsonl(path)

    icl_str = ""

    examples = []
    for d in data:
        premise = f"{d['input']}"
        options = []
        for h in label_list:
            o = {}
            o['premise'] = icl_str + premise
            o['knn_premise'] = premise
            o['hypothesis'] = h
            o['uncond_premise'] = f'{prompt}'
            o['uncond_hypothesis'] = h
            options.append(o)
        label = d['gt_label']
        examples.append({'options' : options, 'label' : label, 'label2synonym': label2synonym, 'label_list': label_list})
    return examples


def load_examples_sst2(path):
    label_list = [' terrible', ' great']
    label_path = "data_eval/fuzzy_verbalizer/sentiment_verbalizer.txt"

    label2synonym = load_label(label_path)
    prompt = " It was"
    icl_str = ""

    data = []
    with open(path) as f:
        for line in f:
            l, s = line.strip().split('\t')
            label = int(l[-1])-3
            if label == 0:
                continue
            d = {}
            d['correct_hypothesis'] = 1 if label > 0 else 0
            d['sentence'] = s
            data.append(d)

    examples = []
    for d in data:
        premise = f"{d['sentence']}{prompt}"
        options = []
        for h in label_list:
            o = {}
            o['premise'] = icl_str + premise
            o['knn_premise'] = premise
            o['hypothesis'] = h
            o['uncond_premise'] = f'{prompt}'
            o['uncond_hypothesis'] = h
            options.append(o)
        label = d['correct_hypothesis']
        examples.append({'options' : options, 'label' : label, 'label2synonym': label2synonym, 'label_list': label_list})
    return examples

def load_examples_hyp(path):
    label2index = [' neutral', ' partisan']
    label_path = "data_eval/fuzzy_verbalizer/hyp_verbalizer.txt"
    label2synonym = load_label(label_path)
    prompt = '\n neutral or partisan? Answer:'
    icl_str = ""

    examples = []
    with open(path) as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            label = int(row['label'])
            summary = row['text'].strip()
            premise = f"{summary}{prompt}"
            options = []
            for h in label2index:
                o = {}
                o['premise'] = icl_str + premise
                o['knn_premise'] = premise
                o['hypothesis'] = h
                o['uncond_premise'] = prompt
                o['uncond_hypothesis'] = h
                options.append(o)
            similarity = float(row["similarity"]) if "similarity" in row else None
            examples.append({'options': options, 'label': label, "sim": similarity, 'label2synonym': label2synonym, 'label_list': label2index})
    return examples


def load_examples_cr(path):
    label2index = [' negative', ' positive']
    label_list = [' terrible', ' great']

    label_path = "data_eval/fuzzy_verbalizer/sentiment_verbalizer.txt"
    label2synonym = load_label(label_path)
    prompt = " It was"
    icl_str = ""
    examples = []
    with open(path) as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            label = int(row['label'])
            summary = row['text']
            premise = f'{summary}{prompt}'
            options = []
            for h in label_list:
                o = {}
                o['premise'] = icl_str + premise
                o['knn_premise'] = premise
                o['hypothesis'] = h.lower()
                o['uncond_premise'] = prompt
                o['uncond_hypothesis'] = h.lower()
                options.append(o)
            similarity = float(row["similarity"]) if "similarity" in row else None
            # print(f"label: {label}")
            # print(f"label_list: {label2index}")
            # print(label_list[label] in label2synonym[label])
            examples.append({'options': options, 'label': label, "sim": similarity, 'label2synonym': label2synonym, 'label_list': label2index})
    return examples

def load_examples_mr(path):
    label2index = [' terrible', ' great']
    label_path = "data_eval/fuzzy_verbalizer/sentiment_verbalizer.txt"
    label2synonym = load_label(label_path)

    prompt = " It was" 
    icl_str = ""

    examples = []
    with open(path) as fp:
        reader = csv.DictReader(fp)
        for i, row in enumerate(reader):
            label = int(row['label'])
            summary = row['text']
            premise = f'{summary}{prompt}'
            # print("premise")
            options = []
            for h in label2index:
                o = {}
                o['premise'] = icl_str + premise
                o['knn_premise'] = premise
                o['hypothesis'] = h.lower()
                o['uncond_premise'] = prompt
                o['uncond_hypothesis'] = h.lower()
                options.append(o)
            similarity = float(row["similarity"]) if "similarity" in row else None
            examples.append({'options': options, 'label': label, "sim": similarity, 'label2synonym': label2synonym, 'label_list': label2index})
    return examples


def load_examples_agn(path):
    topics = [' world', ' sports', ' business', ' science']
    label_path = "data_eval/fuzzy_verbalizer/agn_verbalizer.txt"
    label2synonym = load_label(label_path)
    prompt = " topic:"
    icl_str = ""
    examples = []
    with open(path) as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            label = int(row['Class Index'])-1
            title = row['Title']
            summary = row['Description']
            premise = f"{title} \n {summary}{prompt}"
            options = []
            for h in topics:
                o = {}
                o['premise'] = icl_str + premise
                o['knn_premise'] = premise
                o['hypothesis'] = h
                o['uncond_premise'] = prompt
                o['uncond_hypothesis'] = h
                options.append(o)
            label = label
            examples.append({'options' : options, 'label' : label, 'label2synonym': label2synonym, 'label_list': topics})
    return examples



