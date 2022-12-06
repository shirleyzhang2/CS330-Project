import numpy as np
import json
import os
from absl import logging
import pandas as pd
import re

paraphrase_path = "Tk-Instruct/output/gpt3-paraphrase-tasks-tk-instruct-train-test/predict_results_save_paraphrase.json"
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
# "Tk-Instruct/output/finetuned-v4-paraphrased-instruction/predict_results_save_paraphrase.json" mean std:  14.71674963827671
# "Tk-Instruct/output/notfinetuned-v4-paraphrased-instruction/predict_results_save_paraphrase.json" mean std:  16.57219950559137
# "Tk-Instruct/output/finetuned-v5-paraphrased-instruction/predict_results_save_paraphrase.json" mean std:  25.191888400235214
# "Tk-Instruct/output/notfinetuned-v5-paraphrased-instruction/predict_results_save_paraphrase.json" mean std:  27.238752614299624
# "Tk-Instruct/output/gpt3-paraphrase-tasks-tk-instruct-train-test-finetuned/predict_results_save_paraphrase.json" mean std:  0.4707
# "Tk-Instruct/output/gpt3-paraphrase-tasks-tk-instruct-train-test/predict_results_save_paraphrase.json" mean std:  2.1100