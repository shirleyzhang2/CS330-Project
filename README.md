# CS330-Project

## Data
- classification task data are adapted from natural-instructions-master
- selected tasks from [937, 1388] stored in the `tasks/` folder and on google drive [here](https://drive.google.com/file/d/1HZixR9XLz4X6sQJXIP7viNtPRsx0ZND4/view?usp=share_link)

### Data sample
First 10 textual entailment tasks: `eval/textual_entailment_first10.txt`

All textual entailment tasks: `eval/textual_entailment.txt`

Note: task 463, 464, and 534 are removed because they are non-English.

Read from txt and download tasks to `tasks/` folder: `python utils/download_tasks.py`

## Run Tk-Instruct predictions

Specify whether to use full textual entailment tasks or first 10 textual entailment tasks in `Tk-Instruct/src/ni_dataset.py` 
```
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    # "path": os.path.join(split_dir, "textual_entailment_first10.txt"), 
                    "path": os.path.join(split_dir, "textual_entailment.txt"), 
                    "task_dir": task_dir, 
                    "max_num_instances_per_task": self.config.max_num_instances_per_eval_task,
                    "subset": "test"
                }),
```
Specifiy model configuration (`allenai/tk-instruct-small-def-pos` or `allenai/tk-instruct-base-def-pos`) and output directory in `scripts/eval_tk_instruct.sh`

Run prediction on textual entailment tasks using tkinstruct model: 
```
cd Tk-Instruct/
source scripts/eval_tk_instruct.sh
```

Predictions and metrics are saved at: `Tk-Instruct/output/`

Note: it seems that `tk-instruct-small-def-pos` actually has a better eval exact match accuracy on textual entailment tasks than the larger `allenai/tk-instruct-base-def-pos` model. 

- dependencies:
    - `pip install -r requirements.txt`
    - `pip install -r Tk-Instruct/requirements.txt`

## GPT-3 prompt engineering
#### Paraphrase
Run `paraphrase_prompts.py` with desired arguments. The script uses `paraphrase.prompt` as its template and stores generated prompts by default at `gpt3-results`. The generated results have the following signature:
```
    {
        "orignal_task": "task*.json",
        "action": "Paraphrase",
        "original_prompt": "",
        "generated_prompts": []
    }
```

#### Augment (append choice explanation to the original instruction)
Run `choice_expl_prompts.py` with desired arguments. The script uses `choice_expl.prompt` as its template and stores generated prompts by default at `gpt3-augment-results`. 
The generated results share the above signature.

- dependencies:
    - `pip install -r requirements.txt`
    - `export OPENAI_API_KEY=[YOUR_API_KEY]`

## Run Tk-Instruct with GPT-3 prompts

### Generate new tasks using GPT-3 paraphrased/augmented prompts
#### Paraphrase
Run `generate_gpt3_tasks.py` to fetch paraphrased prompts from `/gpt3-paraphrase-results-tk-instruct-train`, replace original prompts in `/tk-instruct-train-classfication-tasks` 
with paraphrased prompts, store each new task in `/gpt3-paraphrase-tasks-tk-instruct-train`， then return the list of new task names in `eval/textual_entailment_gpt3_paraphrase.txt`.

#### Augment (append choice explanation to the original instruction)
Run `generate_gpt3_tasks.py` to fetch paraphrased prompts from `/gpt3-augment-results`, replace original prompts in `/tasks` 
with paraphrased prompts, store each new task in `/gpt-augment-tasks`， then return the list of new task names in `/eval/textual_entailment_gpt3_augment.txt`.

### Create test references for the new tasks
In `/eval/create_reference_file.py`, update `tasks_dir`, `test_path` and save to `eval/test_references_gpt3_paraphrase.jsonl`.

### Run Tk-Instruct inference using paraphrased/augmented prompts

Specify paraphrased textual entailment tasks in `Tk-Instruct/src/ni_dataset.py` 
```
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    # "path": os.path.join(split_dir, "textual_entailment_first10.txt"), 
                    # "path": os.path.join(split_dir, "textual_entailment.txt"), 
                    "path": os.path.join(split_dir, "textual_entailment_gpt3.txt"), # paraphrase
                    # "path": os.path.join(split_dir, "textual_entailment_gpt3.txt"), # augment
                    "task_dir": task_dir, 
                    "max_num_instances_per_task": self.config.max_num_instances_per_eval_task,
                    "subset": "test"
                }),
```

In `scripts/eval_tk_instruct.sh`, replace 
```
    --task_dir ../tasks
```
with
```
    --task_dir ../gpt3-tasks
```
for paraphrased prompts, or 
```
    --task_dir ../gpt3-augment-tasks
```
for augmented prompts

Specify your own cache location, then run prediction on the paraphrased/augmented textual entailment tasks using tkinstruct model: 
```
cd Tk-Instruct/
source scripts/eval_tk_instruct.sh
```
Predictions and metrics are saved at: `Tk-Instruct/output/` as before.

### Run prompt ensemble
Specify whether to find majority vote or find best paraphrase in `Tk-Instruct/src/ensemble_paraphrased_prompts.py`

Specify input/output paths

Then run
```
cd Tk-Instruct/src
python ensemble_paraphrased_prompts.py
```

### Experiments tracking

Eval set: all 24 English textual entailment tasks (`eval/textual_entailment.txt`) 
Note: task 463, 464, and 534 are removed because they are non-English.

Metrics: Average Exact Match over the 24 English textual entailment tasks

Model: `allenai/tk-instruct-small-def-pos`

| Baseline (original instruction) | Augmented (choice explanation)| Average of 16 paraphrased prompts | Majority vote of 16 paraphrased prompts | Best of 16 paraphrased prompts | Worst of 16 paraphrased prompts | 
| ---- | ---- |  ---- |  ---- | ---- | ---- | 
| 39.5833 | 41.25 | 40.2188 | 40.0833 | 43.8333 | 37.2916 | 







