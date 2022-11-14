import json
from collections import Counter
from compute_metrics import compute_grouped_metrics

predictions_path = "../output/textual_entailment/tk-instruct-small-def-pos/predict_eval_predictions.jsonl"
output_path = "../output/textual_entailment/tk-instruct-small-def-pos/predict_results_ensemble.json"

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