import argparse
import os
import json
from operator import itemgetter
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

os.environ["WANDB_DISABLED"] = "false"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def find_value_std(value, mean, std):
    if std == 0:
        return 0
    if value > mean:
        return (value - mean) / std
    else:
        return - (mean - value) / std

def process_data(args):
    paraphrase_result = json.load(open(args.paraphrase_save_file))

    train_split = []
    test_split = []

    all_pairs = []
    for task, pairs in paraphrase_result.items():
        values = [pair["value"] for pair in pairs]
        mean = np.mean(values)
        std = np.std(values)
        if std <= 3:
            continue
        paraphrases = [pair["paraphrase"] for pair in pairs]
        if args.task_type == "classification":
            max_index, max_value = max(enumerate(values), key=itemgetter(1))
            max_paraphrase = paraphrases[max_index]
            min_index, min_value = min(enumerate(values), key=itemgetter(1))
            min_paraphrase = paraphrases[min_index]
            max_std = (max_value - mean) / std
            min_std = (mean - min_value) / std
            to_add = [{"text": max_paraphrase, "label": 1}, {"text": min_paraphrase, "label": 0}]
            all_pairs.append(to_add)
        elif args.task_type == "regression":
            value_stds = [find_value_std(value, mean, std) for value in values]
            to_add = []
            for value_std, paraphrase in zip(value_stds, paraphrases):
                to_add.append({"text": paraphrase, "label": value_std})
            all_pairs.append(to_add)
    train_count = int(len(all_pairs) * 0.8)
    for i in range(len(all_pairs)):
        pair = all_pairs[i]
        if i > train_count:
            test_split.extend(pair)
        else:
            train_split.extend(pair)

    train_df = pd.DataFrame(train_split)
    test_df = pd.DataFrame(test_split)

    train_df.to_csv(args.train_path)
    test_df.to_csv(args.test_path)

def compute_metrics_classification(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = np.mean(labels == preds)
    return {
        'accuracy': acc,
    }

def compute_metrics_regression(pred):
    predictions, labels = pred
    rmse = mean_squared_error(labels, predictions, squared=False)
    return {"rmse": rmse}

def train(args, prompts_dataset_train, prompts_dataset_test):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)

    tokenized_dataset_train = prompts_dataset_train.map(preprocess_function, batched=True)
    tokenized_dataset_test = prompts_dataset_test.map(preprocess_function, batched=True)
    print(tokenized_dataset_train[0])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    num_labels = 2 if args.task_type == "classification" else 1
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=num_labels)

    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        logging_dir='./logs', 
        logging_steps=5,
        save_strategy="epoch",
        evaluation_strategy="epoch",  
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset_train,
        eval_dataset=tokenized_dataset_test,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_classification if args.task_type == "classification" else compute_metrics_regression,
    )

    trainer.train()

    trainer.evaluate()

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', "--process_data", default=False, type=bool)
    parser.add_argument('-train', "--train_path", default="prompt_quality_data/train_regression_v2.csv")
    parser.add_argument('-test', "--test_path", default="prompt_quality_data/test_regression_v2.csv")
    parser.add_argument('-ps', "--paraphrase_save_file", default="Tk-Instruct/output/gpt3-paraphrase-tasks-tk-instruct-train-train-original-model/predict_results_save_paraphrase_pos.json")
    parser.add_argument("-m", "--model_name", default="roberta-base")
    parser.add_argument("-t", "--task_type", default="regression")
    args = parser.parse_args()

    if args.process_data:
        process_data(args)

    data_files = {"train": args.train_path, "test": args.test_path}
    prompts_dataset_train = load_dataset("csv", data_files=data_files, split="train").shuffle(seed=42)
    prompts_dataset_test = load_dataset("csv", data_files=data_files, split="test").shuffle(seed=42)
    print(prompts_dataset_train[0])

    train(args, prompts_dataset_train, prompts_dataset_test)
