from typing import List
from tqdm import tqdm
import openai
from multiprocessing import Pool
from functools import partial
import math
import argparse
import os
import json
import time
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-i', "--input", help="input task file or dir containing task files")
parser.add_argument('-o', "--output", default="gpt3-results", help="output dir")
parser.add_argument('-t', "--template", default="paraphrase.prompt", help="template file")
parser.add_argument('-a', "--action", default="Paraphrase", help="hint word for GPT-3")
parser.add_argument('-n', "--num_generate", type=int, default=5, help="number of paraphrases to generate")
parser.add_argument("--num_workers", type=int, default=5, help="number of processes used during generation")

# GPT-3 generation hyperparameters
parser.add_argument('--engine', type=str, required=True,
                    choices=['ada',
                             'text-ada-001',
                             'babbage',
                             'text-babbage-001',
                             'curie',
                             'text-curie-001',
                             'davinci',
                             'text-davinci-001',
                             'text-davinci-002'],
                    help='The GPT-3 engine to use.')  # choices are from the smallest to the largest model
parser.add_argument('--max_tokens', type=int, default=40, required=False, help='')
parser.add_argument('--temperature', type=float, default=0.8, required=False, help='')
parser.add_argument('--top_p', type=float, default=0.9, required=False, help='')
parser.add_argument('--frequency_penalty', type=float, default=0.0, required=False, help='')
parser.add_argument('--presence_penalty', type=float, default=0.0, required=False, help='')
parser.add_argument('--stop_tokens', nargs='+', type=str,
                        default=None, required=False, help='Stop tokens for generation')
args = parser.parse_args()

def fill_template(prompt_template_file: str, **prompt_parameter_values) -> str:
    prompt = ''
    with open(prompt_template_file) as prompt_template_file:
        for line in prompt_template_file:
            if line.startswith('#'):
                continue  # ignore comment lines in the template
            prompt += line
    for parameter, value in prompt_parameter_values.items():
        prompt = prompt.replace('{'+parameter+'}', value)
    return prompt

def generate(input_text: str, args, postprocess=True, max_tries=1) -> str:
    """
    text-in-text-out interface to large OpenAI models
    """
    # don't try multiple times if the temperature is 0, because the results will be the same
    if max_tries > 1 and args.temperature == 0:
        max_tries = 1
    # try at most `max_tries` times to get a non-empty output
    for _ in range(max_tries):
        generation_output = openai.Completion.create(engine=args.engine,
                                                     prompt=input_text,
                                                     max_tokens=max(args.max_tokens, len(input_text.split(' '))),
                                                     temperature=args.temperature,
                                                     top_p=args.top_p,
                                                     frequency_penalty=args.frequency_penalty,
                                                     presence_penalty=args.presence_penalty,
                                                     best_of=1,
                                                     stop=args.stop_tokens,
                                                     logprobs=0,  # log probability of top tokens
                                                     )
        # print('raw generation output = ', generation_output)
        # print('='*10)
        generation_output = generation_output['choices'][0]['text']
        generation_output = generation_output.strip()
        if postprocess:
            generation_output = _postprocess_generations(
                generation_output)
        if len(generation_output) > 0:
            break
    return generation_output

def batch_generate(input_texts: List[str], args, postprocess=True, max_tries=1, num_processes=5) -> List[str]:
    """
    Call OpenAI's API in parallel, since each call to the biggest model takes ~1 second to return results
    """
    f = partial(generate, args=args,
                postprocess=postprocess, max_tries=max_tries)
    with Pool(num_processes) as p:
        worker_outputs = list(
            tqdm(p.imap(f, input_texts), total=len(input_texts)))
    return worker_outputs

def _postprocess_generations(generation_output: str) -> str:
    """
    Might output an empty string if generation is not at least one full sentence
    """
    # replace all whitespaces with a single space
    generation_output = ' '.join(generation_output.split())

    # remove extra dialog turns, if any
    if generation_output.find('You: ') > 0:
        generation_output = generation_output[:generation_output.find(
            'You: ')]
    if generation_output.find('They: ') > 0:
        generation_output = generation_output[:generation_output.find(
            'They: ')]

    # delete half sentences
    generation_output = generation_output.strip()
    if len(generation_output) == 0:
        return generation_output

    if generation_output[-1] not in {'.', '!', '?'}:
        last_sentence_end = max(generation_output.find(
            '.'), generation_output.find('!'), generation_output.find('?'))
        if last_sentence_end > 0:
            generation_output = generation_output[:last_sentence_end+1]

    return generation_output


if not os.path.exists(args.output):
    os.makedirs(args.output)

task_paths = []
if os.path.isdir(args.input):
    task_paths = [os.path.join(args.input, filename) for filename in os.listdir(args.input)]
else:
    task_paths = [args.input]

task_names = []
orig_prompts = [] # the Definition in the original task
prompts = [] # templated prompt for gpt3 consisting of task Definition and an action word
for task_file in task_paths:
    task_dict = json.load(open(task_file, 'r'))
    instruction = "\n".join(task_dict['Definition'])
    orig_prompts.append(instruction)
    task_name = os.path.basename(task_file)
    task_names.append(task_name)
    prompt = fill_template(args.template, instruction=instruction, action=args.action)
    prompts.append(prompt)

results = []
for i in range(args.num_generate):
    print(f"Running {i}th round of generation on all tasks")
    num_chunks = len(prompts) // 60 + 1
    chunks = np.array_split(prompts, num_chunks)
    result = []
    for chunk in chunks:
        while True:
            try:
                partial_result = batch_generate(chunk.tolist(), args, num_processes=args.num_workers)
                break
            except openai.error.RateLimitError:
                time.sleep(5)
        result += partial_result
    results.append(result)

for task_name, orig_prompt, gen_prompts in zip(task_names, orig_prompts, [list(result) for result in zip(*results)]):
    data_dict = {
        'orignal_task': task_name,
        'action': args.action,
        'original_prompt': orig_prompt,
        'generated_prompts': gen_prompts
    }
    save_file = os.path.join(args.output, args.action.lower()+'_'+task_name)
    json.dump(data_dict, open(save_file, 'w'), indent=4)