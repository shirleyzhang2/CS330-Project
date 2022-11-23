import random

task_path = "eval/textual_entailment_gpt3_paraphrase.txt"
train_path = "splits/train_tasks.txt"
val_path = "splits/dev_tasks.txt"
test_path = "splits/test_tasks.txt"

# 80/10/10 split
first_split = 0.8
second_split = 0.9

with open(task_path,'r') as f:
    lines = f.readlines()

random.shuffle(lines)
train_line_num = int(len(lines)*first_split)
val_line_num = int(len(lines)*second_split)

with open(train_path, 'w') as f:
    f.writelines(lines[:train_line_num])

with open(val_path, 'w') as f:
    f.writelines(lines[train_line_num:val_line_num])
    
with open(test_path, 'w') as f:
    f.writelines(lines[val_line_num:])