import os
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', "--input", help="input task file or dir containing task files")
parser.add_argument('-o', "--output", default="gpt3-prompts", help="output prompt dir")
args = parser.parse_args()

if not os.path.exists(args.output):
    os.makedirs(args.output)

task_paths = []
if os.path.isdir(args.input):
    task_paths = [os.path.join(args.input, filename) for filename in os.listdir(args.input)]
else:
    task_paths = [args.input]
# print(task_paths)

action =  "Paraphrase" # "Rewrite" #

for task in task_paths:
    task_name = os.path.basename(task)
    prompt_name = task_name.replace('.json', '.prompt')
    print(task_name)
    if not task_name.endswith('.json'):
        raise ValueError
    task_dict = json.load(open(task, 'r'))
    text = "\n".join(task_dict['Definition'])
    print(text)
    
    prompt = f"{action} the following instruction:\n"
    prompt += text
    
    with open(os.path.join(args.output, prompt_name), 'w') as out:
        out.write(prompt)    
