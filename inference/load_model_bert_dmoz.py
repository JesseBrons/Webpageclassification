import re
import os
import sys
import json
import wandb
import torch
import datasets
import evaluate
import numpy as np
import transformers
from pathlib import Path 
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from optimum.bettertransformer import BetterTransformer
from transformers import DataCollatorWithPadding, DefaultDataCollator, AutoModelForSequenceClassification, AutoTokenizer, AutoConfig

MODEL_NAME = str(sys.argv[1])
MODEL_VER = str(sys.argv[2])

set_seed(42)

def prepare_dataset(data_folder, label2id, data_types, max_length, sort_length=False):
    def combine_data(example):
        temp_text = ""
        for data_type in data_types:
            temp_text += example[data_type] + " "
        example["text"] = temp_text
        return example

    def add_length(example):
        example["length"] = len(example["text"])
        return example

    average_length = 0
    dataset = datasets.load_from_disk(data_folder + "dataset/")
    dataset = dataset["test"]
    dataset_encoded = dataset.class_encode_column("category")
    dataset_aligned = dataset_encoded.align_labels_with_mapping(label2id, "category")
    dataset_cleaned = dataset_aligned.map(combine_data)

    if sort_length:
        dataset_cleaned = dataset_cleaned.map(add_length)
        average_length = sum(dataset_cleaned["length"])/len(dataset_cleaned["length"])
        dataset_cleaned = dataset_cleaned.sort("length", reverse=True)
        dataset_cleaned = dataset_cleaned.remove_columns("length")

    dataset = dataset_cleaned.remove_columns(["title", "body"])
    dataset = dataset.rename_column("category", "label")
    return dataset, average_length

def main():
    def preprocess_function_dynamic(examples):
        return tokenizer(examples["text"], padding=True, truncation=True, max_length=hps["max_length"])

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, padding='max_length', max_length=hps["max_length"])

    models = {"bert": "bert-base-uncased", "distilbert": "distilbert-base-uncased", "tinybert": "huawei-noah/TinyBERT_General_4L_312D"} 

    hps = {
        "dynamic_padding": False,
        "better_transformer": False,
        "batch_size": 128,
        "max_length": 128,
        "data_types": ["title", "body"],
        "model_name": models[MODEL_NAME],
        "mixed_precision": "fp32",
    }

    data_folder = str(sys.argv[3])

    api = wandb.Api()
    model_artifact = api.artifact("DMOZ-classification/model_" + MODEL_NAME.upper() + "_DMOZ:" + MODEL_VER)
    logged_by = model_artifact.logged_by().group
    model_dir = model_artifact.download(data_folder + "artifacts/")
    
    wandb.finish()

    accelerator = Accelerator(log_with="wandb")

    accelerator.init_trackers(
        project_name="DMOZ-classification",
        config=hps,
        init_kwargs={"wandb": {
            "name": "INFERENCE_" + logged_by,
            "job_type": "inference",
            "group": logged_by, 
            "tags": [MODEL_NAME.upper(), "DMOZ"],
            }
        },
    )

    id2label = {0: "Arts", 1: "Business", 2: "Computers", 3: "Health", 4: "Home", 5: "News", 6: "Recreation", 7: "Reference", 8: "Science", 9: "Shopping", 10: "Society", 11: "Sports", 12: "Games"}
    label2id = {v: k for k, v in id2label.items()}
    labels = label2id.keys()
 
    dataset, average_length = prepare_dataset(data_folder, label2id, hps["data_types"], hps["max_length"], sort_length=hps["dynamic_padding"])

    if hps["dynamic_padding"]:
        accelerator.log({"Average length": average_length})

    tokenizer = AutoTokenizer.from_pretrained(hps["model_name"])

    model = AutoModelForSequenceClassification.from_pretrained(hps["model_name"], num_labels=len(labels))
    model.load_state_dict(torch.load(model_dir + "model.pt"))
    if hps["better_transformer"]:
        model = BetterTransformer.transform(model)
    
    if hps["dynamic_padding"]:
        tokenized_data = dataset.map(preprocess_function_dynamic, batched=True)
        data_collator = DataCollatorWithPadding(tokenizer, padding="longest")
    else:
        tokenized_data = dataset.map(preprocess_function, batched=True)
        data_collator = DefaultDataCollator()
    tokenized_data = tokenized_data.remove_columns("text")

    test_dataloader = DataLoader(tokenized_data, batch_size=hps["batch_size"], collate_fn=data_collator, drop_last=True)

    test_dataloader, model = accelerator.prepare(test_dataloader, model)

    inference_start, inference_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = np.zeros((len(test_dataloader), 1))

    metric_acc = evaluate.load("accuracy")
    metric_f1 = evaluate.load("f1")
    metric_prec = evaluate.load("precision") 
    metric_recall = evaluate.load("recall") 
    
    model.eval()
    #GPU Warmup
    with torch.no_grad():
        for i in range(10):
            dummy_batch = next(iter(test_dataloader))
            _ = model(**dummy_batch)

    #Inference
    inference_start.record()
    with torch.no_grad():
        for idx, batch in enumerate(test_dataloader):
            starter.record()
            outputs = model(**batch)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[idx] = curr_time
            predictions = outputs.logits.argmax(dim=-1)
            predictions, references = accelerator.gather_for_metrics(
                (predictions, batch["labels"])
            )
            metric_acc.add_batch(predictions=predictions, references=references)
            metric_f1.add_batch(predictions=predictions, references=references)
            metric_prec.add_batch(predictions=predictions, references=references)
            metric_recall.add_batch(predictions=predictions, references=references)
            
    inference_end.record()
    torch.cuda.synchronize()
    inference_time = inference_start.elapsed_time(inference_end)

    mean_syn = np.sum(timings) / (len(test_dataloader) * hps["batch_size"])
    std_syn = np.std(timings) 

    accuracy = metric_acc.compute()
    f1 = metric_f1.compute(average="weighted")
    precision = metric_prec.compute(average="weighted")
    recall = metric_recall.compute(average="weighted")

    accelerator.log({"f1_weighted": f1["f1"], "precision": precision["precision"], "recall": recall["recall"], "accuracy": accuracy["accuracy"], "inference_time": inference_time, "mean_time": mean_syn, "std_time": std_syn})

if __name__ == "__main__":
    main()
