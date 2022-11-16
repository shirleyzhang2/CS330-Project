import argparse
import json
import random
from collections import Counter
import numpy as np
from compute_metrics import compute_grouped_metrics

parser = argparse.ArgumentParser()
parser.add_argument('-p', "--predictions_path", default="../output/textual_entailment/tk-instruct-small-def-pos/predict_eval_predictions_all_paraphrases.jsonl", help="input path to load predictions jsonl for all paraphrased prompts")
parser.add_argument('-m', "--metrics_path", default="../output/textual_entailment/tk-instruct-small-def-pos/predict_results_all_paraphrases.json", help="input path to load metrics json for all paraphrased prompts")
parser.add_argument('-o', "--output_path", default="../output/textual_entailment/tk-instruct-small-def-pos/predict_results_ensemble.json", help="output path to save metrics, e.g. best, ensemble, k_random_avg")
parser.add_argument('-k', "--k", default=4, help="number of paraphrased prompts to sample")
args = parser.parse_args()

def find_majority_vote_metrics(predictions_path, output_path):
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
        c = Counter(preds)
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

def find_best_paraphrase_metrics(metrics_path, output_path):
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
    best_metrics_by_task = {}
    exact_match_total = []
    for task, values in metrics_by_task.items():
        max_value = max(values)
        best_metrics_by_task[task] = max_value
        exact_match_total.append(max_value)
    
    exact_match = sum(exact_match_total) / len(exact_match_total)
    best_metrics_by_task["exact_match"] = exact_match

    with open(output_path, 'w') as output_file:
        json.dump(best_metrics_by_task, output_file, indent=4)

def find_k_paraphrase_avg_metrics(metrics_path, output_path, k=1):
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
    find_majority_vote_metrics(args.predictions_path, args.output_path)

    # find best paraphrase
    # find_best_paraphrase_metrics(args.metrics_path, args.output_path)

    # find average of k randomly sampled paraphrase
    # find_k_paraphrase_avg_metrics(args.metrics_path, args.output_path, k=args.k)
