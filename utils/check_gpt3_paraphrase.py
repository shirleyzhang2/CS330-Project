import os
import json

for f in os.listdir('gpt3-paraphrase-results-tk-instruct-train/'):
    dic = json.load(open('gpt3-paraphrase-results-tk-instruct-train/'+f))
    if len(dic['generated_prompts']) < 32:
        print(f)