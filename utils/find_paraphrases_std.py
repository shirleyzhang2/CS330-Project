import numpy as np
import json
import os
from absl import logging
import pandas as pd
import re

paraphrase_path = "Tk-Instruct/output/gpt3-paraphrase-tasks-tk-instruct-finetuned-v2/predict_results_save_paraphrase.json"
paraphrase_result = json.load(open(paraphrase_path))
std_lst = []
for task, pairs in paraphrase_result.items():
    values = [pair["value"] for pair in pairs]
    paraphrases = [pair["paraphrase"] for pair in pairs]
    mean = np.mean(values)
    std = np.std(values)
    std_lst.append(std)

print("mean std: ", np.mean(std_lst)) 
# "Tk-Instruct/output/gpt3-paraphrase-tasks-tk-instruct-finetuned-v2/predict_results_save_paraphrase.json"  mean std: 1.4318980442161025
# "Tk-Instruct/output/gpt3-paraphrase-tasks-tk-instruct-notfinetuned-v2/predict_results_save_paraphrase.json" mean std: 5.113575891515915

