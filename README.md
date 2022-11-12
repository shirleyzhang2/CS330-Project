# CS330-Project

First 10 textual entailment tasks: `eval/textual_entailment_first10.txt`

Read from txt and download tasks to `tasks/` folder: `python utils/download_tasks.py`

Specifiy model configuration (currently we are using the smallest `allenai/tk-instruct-small-def-pos` model) and run prediction on the first 10 textual entailment tasks using tkinstruct model: 
```
cd Tk-Instruct/
source scripts/eval_tk_instruct.sh
```

Predictions are saved at: `Tk-Instruct/output/predict_eval_predictions.jsonl`

Metrics are saved at: `Tk-Instruct/output/predict_results.json`
