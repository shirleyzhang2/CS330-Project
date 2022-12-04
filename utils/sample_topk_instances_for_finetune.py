import os
import json
import random

topk = 8
input_directory = "gpt3-paraphrase-tasks-tk-instruct-test"
output_directory = f"gpt3-paraphrase-tasks-tk-instruct-test-top{topk}"

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

for filename in os.listdir(input_directory):
    input_f = os.path.join(input_directory, filename)
    output_f = os.path.join(output_directory, filename)
    if not os.path.isfile(input_f):
        continue
    if not input_f.endswith(".json"):
        continue

    data = json.load(open(input_f))
    all_instances = data["Instances"]
    sampled_instances = random.sample(all_instances, topk)
    data["Instances"] = sampled_instances

    json_object = json.dumps(data, indent=4)
    with open(output_f, "w") as outfile:
        outfile.write(json_object)