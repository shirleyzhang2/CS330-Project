import random

task_path = "eval/gpt3-paraphrase-results-tk-instruct-train.txt"
train_path = "splits/train_tasks_v1.txt"
eval_path = "splits/dev_tasks_v1.txt"

# 80/20 split train and validation
first_split = 0.8

with open(task_path,'r') as f:
    lines = f.readlines()

random.shuffle(lines)
train_line_num = int(len(lines)*first_split)

with open(train_path, 'w') as f:
    f.writelines(lines[:train_line_num])
    
with open(eval_path, 'w') as f:
    f.writelines(lines[train_line_num:])