import numpy as np
import json
import os
from absl import logging
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import re

from train_prompt_quality_classifier import find_value_std


paraphrase_path = "Tk-Instruct/output/gpt3-paraphrase-tasks-tk-instruct-train/predict_results_save_paraphrase.json"
original_path = "Tk-Instruct/output/tk_instruct_train/tk-instruct-small-def-pos/predict_results_original_instruction.json"
task_dir = "tk-instruct-train-classfication-tasks/"
paraphrase_similarity_path = "Tk-Instruct/output/gpt3-paraphrase-tasks-tk-instruct-train/predict_results_save_paraphrase_similarity.json"

module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
model = hub.load(module_url)
print ("module %s loaded" % module_url)
def embed(input):
  return model(input)

def read_instruction_from_task_file(filename):
    data = json.load(open(filename))
    instruction = data["Definition"][0]
    return instruction

def process_filename(metric_name):
    task_name = metric_name.removeprefix("predict_exact_match_for_")
    task_name += ".json"
    task_name = os.path.join(task_dir, task_name)
    return task_name

metrics = json.load(open(original_path))
task_to_original_instruct = {}
for key, value in metrics.items():
    if not key.startswith("predict_exact_match_for_task"):
        continue
    filename = process_filename(key)
    original_instruction = read_instruction_from_task_file(filename)
    task_to_original_instruct[key] = original_instruction

paraphrase_result = json.load(open(paraphrase_path))
paraphrase_similarity = paraphrase_result.copy()
similarity_lst = []
value_std_lst = []
for task, pairs in paraphrase_result.items():
    original_instruction = task_to_original_instruct[task]
    original_instruction_embedding = np.array(embed([original_instruction])).tolist()[0]
    values = [pair["value"] for pair in pairs]
    paraphrases = [pair["paraphrase"] for pair in pairs]
    paraphrase_embeddings = embed(paraphrases)
    mean = np.mean(values)
    std = np.std(values)
    value_stds = [find_value_std(value, mean, std) for value in values]
    paraphrase_similarity[task] = []
    for i, paraphrase_embedding in enumerate(np.array(paraphrase_embeddings).tolist()):
        paraphrase = paraphrases[i]
        value = values[i]
        value_std = value_stds[i]
        similarity = np.dot(original_instruction_embedding, paraphrase_embedding)
        paraphrase_similarity[task].append({
            "paraphrase": paraphrase,
            "value": value, 
            "value_std": value_std,
            "similarity": similarity
        })
        similarity_lst.append(similarity)
        value_std_lst.append(value_std)

print("corr: ", np.corrcoef(similarity_lst, value_std_lst)) # 0.05815013

with open(paraphrase_similarity_path, 'w') as output_file:
    json.dump(paraphrase_similarity, output_file, indent=4)
