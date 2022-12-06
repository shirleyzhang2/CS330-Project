import argparse
import os
import json
import random
from collections import Counter
from operator import itemgetter
import numpy as np
from compute_metrics import compute_grouped_metrics

parser = argparse.ArgumentParser()
parser.add_argument('-p', "--predictions_path", default="../output/textual_entailment/tk-instruct-small-def-pos/predict_eval_predictions_all_paraphrases.jsonl", help="input path to load predictions jsonl for all paraphrased prompts")
parser.add_argument('-m', "--metrics_path", default="../output/gpt3-paraphrase-tasks-tk-instruct-train-train-original-model/predict_results_all_paraphrases.json", help="input path to load metrics json for all paraphrased prompts")
parser.add_argument('-o', "--output_path", default="../output/gpt3-paraphrase-tasks-tk-instruct-train-train-original-model/predict_results_best.json", help="output path to save metrics, e.g. best, ensemble, k_random_avg")
parser.add_argument('-k', "--k", default=32, help="number of paraphrased prompts to sample")
parser.add_argument('-ps', "--paraphrase_save_file", default="../output/gpt3-paraphrase-tasks-tk-instruct-train-train-original-model/predict_results_save_paraphrase_pos.json", help="output path to save metrics, e.g. best, ensemble, k_random_avg")
parser.add_argument('-t', "--task_dir", default="../../gpt3-paraphrase-tasks-tk-instruct-train", help="output path to save metrics, e.g. best, ensemble, k_random_avg")
args = parser.parse_args()

def read_instruction_from_task_file(filename):
    data = json.load(open(filename))
    instruction = data["Definition"][0]
    return instruction

def read_instruction_and_pos_from_task_file(filename):
    task_dict = json.load(open(filename))
    instruction = task_dict["Definition"][0]
    if not instruction.endswith('.'):
        instruction += '.' # prevent gpt3 generating a dot
    result = "Task instruction: " + instruction

    pos_examples = task_dict["Positive Examples"]
    pos_examples_by_class = {}
    for pos_example in pos_examples:
        example_output = pos_example["output"]
        if example_output in pos_examples_by_class:
            pos_examples_by_class[example_output].append(pos_example)
        else:
            pos_examples_by_class[example_output] = [pos_example]
    first_examples = []
    num_examples = 1
    for _, examples in pos_examples_by_class.items():
        first_example = examples[0] # select 1 example per class
        first_examples.append(first_example)
        result += f"({num_examples})\n"
        example_input = first_example["input"]
        result += "Input: " + example_input + "\n"
        example_output = first_example["output"]
        result += "Output: " + example_output + "\n"
        # example_explanation = first_example["explanation"]
        # result += "Explanation: " + example_explanation + "\n"
    # print(result)
    return result

def read_classes_from_task_file(filename):
    task_dict = json.load(open(filename))
    pos_examples = task_dict["Positive Examples"]
    classes = list(set([pos_example["output"] for pos_example in pos_examples]))
    return classes

def find_random_class_acc_from_task_file(filename):
    task_dict = json.load(open(filename))
    pos_examples = task_dict["Positive Examples"]
    classes = list(set([pos_example["output"] for pos_example in pos_examples]))
    instances = task_dict["Instances"]
    correct_sum = 0
    for instance in instances[:100]:
        sampled_class = random.choice(classes)
        gt_class = instance["output"][0]
        # print(sampled_class, gt_class)
        correct_sum += gt_class == sampled_class
    acc = correct_sum / 100 # 100 instances
    return acc 


def process_filename(metric_name):
    task_name = metric_name.removeprefix("predict_exact_match_for_")
    task_name += ".json"
    # text-davinci-002_paraphrase.prompt_task1344_glue_entailment_classification.json
    # task_name = "text-davinci-002_paraphrase.prompt_" + task_name 
    task_name = os.path.join(args.task_dir, task_name)
    return task_name

def find_majority_vote_metrics(predictions_path, output_path, k):
    with open(predictions_path, 'r') as json_file:
        json_list = list(json_file)

    predictions_by_instanceid = {}
    for entry in json_list:
        d = json.loads(entry)
        instance_id = d["Instance"]["id"]
        if instance_id not in predictions_by_instanceid:
                predictions_by_instanceid[instance_id] = [d]
        else:
                predictions_by_instanceid[instance_id].append(d)
    predictions = []
    references = []
    groups = []
    for instance_id, ds in predictions_by_instanceid.items():
        reference = ds[0]["Instance"]["output"]
        task = ds[0]["Task"].split("_gpt")[0]
        preds = [d["Prediction"] for d in ds]
        #downsample
        sampled_preds = random.sample(preds, k)
        c = Counter(sampled_preds)
        prediction, _ = c.most_common()[0]
        predictions.append(prediction)
        references.append(reference)
        groups.append(task)


    metrics = compute_grouped_metrics(predictions, references, groups, xlingual=False)
    exact_match_total = []
    for key, value in metrics.items():
            if key.startswith("exact_match"):
                exact_match_total.append(value)
    exact_match = sum(exact_match_total) / len(exact_match_total)
    metrics["exact_match"] = exact_match

    with open(output_path, 'w') as output_file:
        json.dump(metrics, output_file, indent=4)

def find_best_metrics_by_task(metrics_by_task, k, best=True):
    best_metrics_by_task = {}
    best_paraphrase_by_task = {}
    exact_match_total = []
    for task, pairs in metrics_by_task.items():
        values = [pair["value"] for pair in pairs]
        paraphrases = [pair["paraphrase"] for pair in pairs]
        # downsample
        if k > len(values):
            print(f"{task} has fewer than {k} paraphrases")
        min_k = min(int(k), len(values))
        
        sampled_values = random.sample(values, min_k)
        if best:
            index, selected_value = max(enumerate(sampled_values), key=itemgetter(1))
        else:
            index, selected_value = min(enumerate(sampled_values), key=itemgetter(1))
        # if paraphrase_save_file:
        #     selected_paraphrase_name = paraphrases[index]
        #     selected_paraphrase_filename = process_filename(selected_paraphrase_name)
        #     selected_paraphrase = read_instruction_from_task_file(selected_paraphrase_filename)
        #     best_paraphrase_by_task[task] = selected_paraphrase
        best_metrics_by_task[task] = selected_value
        exact_match_total.append(selected_value)
    
    exact_match = sum(exact_match_total) / len(exact_match_total)
    best_metrics_by_task["exact_match"] = exact_match
    print("best_metrics_by_task['exact_match']: ", best_metrics_by_task["exact_match"])
    return best_metrics_by_task

def find_best_metrics_by_task_allk(metrics_by_task, best=True):
    for k in [1, 2, 4, 8, 16, 32]:
        best_metrics_by_task = {}
        best_paraphrase_by_task = {}
        exact_match_total = []
        for task, pairs in metrics_by_task.items():
            values = [pair["value"] for pair in pairs]
            paraphrases = [pair["paraphrase"] for pair in pairs]
            # downsample
            if k > len(values):
                print(f"{task} has fewer than {k} paraphrases")
            min_k = min(int(k), len(values))
            
            sampled_values = random.sample(values, min_k)
            if best:
                index, selected_value = max(enumerate(sampled_values), key=itemgetter(1))
            else:
                index, selected_value = min(enumerate(sampled_values), key=itemgetter(1))
            best_metrics_by_task[task] = selected_value
            exact_match_total.append(selected_value)
        
        exact_match = sum(exact_match_total) / len(exact_match_total)
        best_metrics_by_task["exact_match"] = exact_match
        print(f"best exact match at {k}: ", best_metrics_by_task["exact_match"])

def find_paraphrase_metrics(metrics_path, output_path, k, best=True, paraphrase_save_file=""):
    metrics = json.load(open(metrics_path))
    # group by tasks
    metrics_by_task = {}
    for key, value in metrics.items():
        if not key.startswith("predict_exact_match_for_task"):
            continue
        task = key.split("_gpt")[0]
        if paraphrase_save_file:
            selected_paraphrase_filename = process_filename(key)
            selected_paraphrase = read_instruction_from_task_file(selected_paraphrase_filename)
            # selected_paraphrase = read_instruction_and_pos_from_task_file(selected_paraphrase_filename)
        else:
            selected_paraphrase = key
        pair = {
            "value": value,
            "paraphrase": selected_paraphrase
        }
        if task in metrics_by_task:
            metrics_by_task[task].append(pair)
        else:
            metrics_by_task[task] = [pair]
    if paraphrase_save_file:
        with open(paraphrase_save_file, 'w') as output_file:
            json.dump(metrics_by_task, output_file, indent=4)
    
    best_metrics_by_task = find_best_metrics_by_task(metrics_by_task, k)
    # find_best_metrics_by_task_allk(metrics_by_task)
    
    with open(output_path, 'w') as output_file:
        json.dump(best_metrics_by_task, output_file, indent=4)

def find_best_random_label(metrics_path):
    # assign random label for each instance in each paraphrased task
    metrics = json.load(open(metrics_path))
    # group by tasks
    metrics_by_task = {}
    for key, value in metrics.items():
        if not key.startswith("predict_exact_match_for_task"):
            continue
        task = key.split("_gpt")[0]
        filename = process_filename(key)
        acc = find_random_class_acc_from_task_file(filename)
        # selected_paraphrase = read_instruction_from_task_file(selected_paraphrase_filename)
        pair = {
            "value": acc,
            "paraphrase": key
        }
        if task in metrics_by_task:
            metrics_by_task[task].append(pair)
        else:
            metrics_by_task[task] = [pair]

    find_best_metrics_by_task_allk(metrics_by_task)
    

def find_k_paraphrase_avg_metrics(metrics_path, output_path, k):
    metrics = json.load(open(metrics_path))
    # group by tasks
    metrics_by_task = {}
    for key, value in metrics.items():
        if not key.startswith("predict_exact_match_for_task"):
            continue
        task = key.split("_gpt")[0]
        if task in metrics_by_task:
            metrics_by_task[task].append(value)
        else:
            metrics_by_task[task] = [value]
    avg_metrics_by_task = {}
    exact_match_total = []
    for task, values in metrics_by_task.items():
        # downsample
        sampled_values = random.sample(values, k) # sample without replacement
        avg_value = np.mean(sampled_values)
        avg_metrics_by_task[task] = avg_value
        exact_match_total.append(avg_value)
    
    exact_match = sum(exact_match_total) / len(exact_match_total)
    avg_metrics_by_task["exact_match"] = exact_match

    with open(output_path, 'w') as output_file:
        json.dump(avg_metrics_by_task, output_file, indent=4)
        

if __name__=="__main__":
    # find majority vote
    #find_majority_vote_metrics(args.predictions_path, args.output_path, args.k)

    # find paraphrase metrics (best or worth)
    find_paraphrase_metrics(args.metrics_path, args.output_path, args.k, best=True, paraphrase_save_file=args.paraphrase_save_file)
    # find_paraphrase_metrics(args.metrics_path, 32, best=True)
    # find_best_random_label(args.metrics_path)

    # find average of k randomly sampled paraphrase
    # find_k_paraphrase_avg_metrics(args.metrics_path, args.output_path, k=args.k)
