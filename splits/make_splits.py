import random

task_path = "eval/textual_entailment_gpt3_paraphrase.txt"
train_path = "splits/train_tasks.txt"
test_path = "splits/test_tasks.txt"
split_ratio = 0.2

with open(task_path,'r') as f:
    lines = f.readlines()

random.shuffle(lines)
num_lines = int(len(lines)*split_ratio)

with open(train_path, 'w') as f:
    f.writelines(lines[:num_lines])
with open(test_path, 'w') as f:
    f.writelines(lines[num_lines:])