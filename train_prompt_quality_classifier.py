import argparse
import os
import json
from operator import itemgetter
import pandas as pd
import numpy as np

from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def process_data(args):
    paraphrase_result = json.load(open(args.paraphrase_save_file))

    train_split = []
    test_split = []

    train_count = int(len(paraphrase_result.items()) * 0.8)
    i = 0
    for task, pairs in paraphrase_result.items():
        values = [pair["value"] for pair in pairs]
        paraphrases = [pair["paraphrase"] for pair in pairs]
        max_index, max_value = max(enumerate(values), key=itemgetter(1))
        max_paraphrase = paraphrases[max_index]
        min_index, min_value = min(enumerate(values), key=itemgetter(1))
        min_paraphrase = paraphrases[min_index]
        to_add = [{"text": max_paraphrase, "label": 1}, {"text": min_paraphrase, "label": 0}]
        i += 1
        if i > train_count:
            test_split.extend(to_add)
        else:
            train_split.extend(to_add)

    train_df = pd.DataFrame(train_split)
    test_df = pd.DataFrame(test_split)

    train_df.to_csv(args.train_path)
    test_df.to_csv(args.test_path)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = np.mean(labels == preds)
    return {
        'accuracy': acc,
    }

def train(args, prompts_dataset_train, prompts_dataset_test):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)

    tokenized_dataset_train = prompts_dataset_train.map(preprocess_function, batched=True)
    tokenized_dataset_test = prompts_dataset_test.map(preprocess_function, batched=True)
    print(tokenized_dataset_train[0])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        logging_dir='./logs', 
        logging_steps=5,
        save_steps=5,
        evaluation_strategy="steps",  
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset_train,
        eval_dataset=tokenized_dataset_test,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    trainer.evaluate()

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', "--process_data", default=False, type=bool)
    parser.add_argument('-train', "--train_path", default="prompt_quality_data/train.csv")
    parser.add_argument('-test', "--test_path", default="prompt_quality_data/test.csv")
    parser.add_argument('-ps', "--paraphrase_save_file", default="Tk-Instruct/output/tk_instruct_train/tk-instruct-small-def-pos/predict_results_save_paraphrase.json")
    parser.add_argument("-m", "--model_name", default="distilbert-base-cased")
    args = parser.parse_args()

    if args.process_data:
        process_data(args)

    data_files = {"train": args.train_path, "test": args.test_path}
    prompts_dataset_train = load_dataset("csv", data_files=data_files, split="train").shuffle(seed=42)
    prompts_dataset_test = load_dataset("csv", data_files=data_files, split="test").shuffle(seed=42)
    print(prompts_dataset_train[0])

    train(args, prompts_dataset_train, prompts_dataset_test)
