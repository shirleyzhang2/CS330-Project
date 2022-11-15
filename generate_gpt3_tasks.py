import os
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', "--input", default="tasks", help="input task file or dir containing task files")
parser.add_argument('-p', "--prompt", default="gpt3-augment-results", help="input paraphrased prompt file or dir containing paraphrased prompt files")
parser.add_argument('-l', "--list", default="eval/textual_entailment_gpt3.txt", help="output file for list of paraphrased tasks")
parser.add_argument('-o', "--output", default="gpt3-augment-tasks", help="output dir for tasks with paraphrased prompts")
args = parser.parse_args()

if not os.path.exists(args.output):
    os.makedirs(args.output)

prompt_paths = []
if os.path.isdir(args.prompt):
    prompt_paths = [os.path.join(args.prompt, filename) for filename in os.listdir(args.prompt)]
else:
    prompt_paths = [args.prompt]


for task in prompt_paths:
    paraphrased_prompt_dict = json.load(open(task, 'r'))
    task_name = paraphrased_prompt_dict['orignal_task']
    generated_prompts = paraphrased_prompt_dict['generated_prompts']

    task_dir = os.path.join(args.input, task_name)
    task_dict = json.load(open(task_dir, 'r', encoding="utf-8"))
    
    if isinstance(generated_prompt, str):
        generated_prompts = [generated_prompts]

    for i, generated_prompt in enumerate(generated_prompts):
        task_dict['Definition'] = [generated_prompt]
        save_file = os.path.join(args.output, task_name[:-5] + '_' + 'gpt3_' + str(i)+'.json')
        json.dump(task_dict, open(save_file, 'w'), indent=4)


with open(args.list, "w") as a:
    for path, subdirs, files in os.walk(args.output):
       for filename in files:
         a.write(str(filename)[:-5] + '\n')