import random

task_path = "eval/gpt3-paraphrase-results-tk-instruct-train.txt"
train_path = "splits/train_tasks_v2.txt"
test_path = "splits/test_tasks_v2.txt"

# 80/20 split (no validation)
first_split = 0.8

with open(task_path,'r') as f:
    lines = f.readlines()

#random.shuffle(lines)
train_line_num = int(len(lines)*first_split)

with open(train_path, 'w') as f:
    f.writelines(lines[:train_line_num])
    
with open(test_path, 'w') as f:
    f.writelines(lines[train_line_num:])