import os
import json

# all tasks path
all_tasks_path = "gpt3-paraphrase-tasks-tk-instruct"
test_tasks_save_path = "gpt3-paraphrase-tasks-tk-instruct-test"
test_tasks_txt_path = "splits/test_tasks_v2.txt"

train_tasks_save_path = "gpt3-paraphrase-tasks-tk-instruct-train"
train_tasks_txt_path = "splits/train_tasks_v2.txt"

if not os.path.exists(test_tasks_save_path):
    os.makedirs(test_tasks_save_path)
    
if not os.path.exists(train_tasks_save_path):
    os.makedirs(train_tasks_save_path)    

all_tasks_list = [os.path.splitext(filename)[0] for filename in os.listdir(all_tasks_path)]

test_tasks = open(test_tasks_txt_path, "r")
test_data = test_tasks.read()
test_data_list = test_data.split("\n")

train_tasks = open(train_tasks_txt_path, "r")
train_data = train_tasks.read()
train_data_list = train_data.split("\n")

for task in all_tasks_list:
    if task in test_data_list:
        task_dir = os.path.join(all_tasks_path, task+'.json')
        task_dict = json.load(open(task_dir, 'r', encoding="utf-8"))
        save_file = os.path.join(test_tasks_save_path, task+'.json')
        json.dump(task_dict, open(save_file, 'w'), indent=4)
        
    if task in train_data_list:
        task_dir = os.path.join(all_tasks_path, task+'.json')
        task_dict = json.load(open(task_dir, 'r', encoding="utf-8"))
        save_file = os.path.join(train_tasks_save_path, task+'.json')
        json.dump(task_dict, open(save_file, 'w'), indent=4)
        