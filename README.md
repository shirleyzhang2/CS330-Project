# CS330-Project

## Data
- classification task data are adapted from natural-instructions-master
- selected tasks from [937, 1388] stored in the `tasks/` folder and on google drive [here](https://drive.google.com/file/d/1HZixR9XLz4X6sQJXIP7viNtPRsx0ZND4/view?usp=share_link)

### Data sample
First 10 textual entailment tasks: `eval/textual_entailment_first10.txt`
All textual entailment tasks: `eval/textual_entailment.txt`

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

## GPT-3 prompt engineering
Run `paraphrase_prompts.py` with desired arguments. The script uses `paraphrase.prompt` as its template and stores generated prompts by default at `gpt3-results`. The generated results have the following signature:
```
    {
        "orignal_task": "task*.json",
        "action": "Paraphrase",
        "original_prompt": "",
        "generated_prompts": []
    }
```

- dependencies:
    - `pip install -r requirements.txt`
    - `export OPENAI_API_KEY=[YOUR_API_KEY]`

