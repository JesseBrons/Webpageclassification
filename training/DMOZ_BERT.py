import re
import os
import sys
import time
import json
import torch
import wandb
import random
import datasets
import evaluate
import numpy as np
import transformers
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DefaultDataCollator, AutoModelForSequenceClassification

set_seed(42)

MODEL_NAME = str(sys.argv[1])

def prepare_dataset(data_folder, label2id, data_types):
    def combine_data(example):
        temp_text = ""
        for data_type in data_types:
            temp_text += example[data_type] + " "
        example["text"] = temp_text
        return example
    
    dataset = datasets.load_from_disk(data_folder + "dataset/")
    dataset = dataset["train"]
    dataset_encoded = dataset.class_encode_column("category")
    dataset_aligned = dataset_encoded.align_labels_with_mapping(label2id, "category")
    dataset = dataset_aligned.map(combine_data, remove_columns=["title", "body"])
    dataset = dataset.rename_column("category", "label")

    return dataset

def main():
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, padding='max_length', max_length=hps["max_length"], return_tensors='pt')

    
    models = {"bert": "bert-base-uncased", "distilbert": "distilbert-base-uncased", "tinybert": "huawei-noah/TinyBERT_General_4L_312D"} 

    hps = {
        "batch_size": 32,
        "gradient_accumulation_steps": 2,
        "learning_rate": 2e-5, 
        "data_types": ["title", "body"],
        "model_name": models[MODEL_NAME],
        "num_epochs": 3,
        "max_length": 256,
        "weight_decay": 0.01,
        "num_warmup_steps": 0.2,
        "mixed_precision": "no",
        "split_batches": True,
    }
    
    wandb_id = wandb.util.generate_id()

    accelerator = Accelerator(log_with="wandb", gradient_accumulation_steps=hps["gradient_accumulation_steps"], split_batches=hps["split_batches"], mixed_precision=hps["mixed_precision"])

    accelerator.init_trackers(
        project_name="DMOZ-classification",
        config=hps,
        init_kwargs={"wandb": {
            "name": MODEL_NAME.upper() + "_DMOZ_" + str(wandb_id),
            "job_type": "training",
            "group": str(wandb_id),
            "tags": [MODEL_NAME.upper(), "DMOZ"],
            }
        },
    )

    data_folder = str(sys.argv[2])

    id2label = {0: "Arts", 1: "Business", 2: "Computers", 3: "Health", 4: "Home", 5: "News", 6: "Recreation", 7: "Reference", 8: "Science", 9: "Shopping", 10: "Society", 11: "Sports", 12: "Games"}
    label2id = {v: k for k, v in id2label.items()}
    labels = label2id.keys()

    dataset = prepare_dataset(data_folder, label2id, hps["data_types"])
  
    tokenizer = AutoTokenizer.from_pretrained(hps["model_name"])
    data_collator = DefaultDataCollator()

    tokenized_data = dataset.map(preprocess_function, batched=True)
    tokenized_data = tokenized_data.remove_columns("text")

    train_dataloader = DataLoader(
        tokenized_data, 
        shuffle=True, 
        batch_size=hps["batch_size"], 
        collate_fn=data_collator, 
        drop_last=True,
    )
    
    model = AutoModelForSequenceClassification.from_pretrained(
        hps["model_name"], 
        num_labels=len(labels), 
        id2label=id2label, label2id=label2id,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=(hps["learning_rate"] * accelerator.num_processes),
        weight_decay=hps["weight_decay"],
        eps=1e-8,
    )

    num_training_steps = hps["num_epochs"] * len(tokenized_data)
    num_warmup_steps = int(hps["num_warmup_steps"] * len(train_dataloader))
    lr_scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
    )

    train_dataloader, model, optimizer, lr_scheduler = accelerator.prepare(train_dataloader, model, optimizer, lr_scheduler)

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    accuracy = evaluate.load("accuracy")

    model.train()
    starter.record()
    for epoch in range(hps["num_epochs"]):
        for idx, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                logits = outputs.logits
                accelerator.backward(loss)

                predictions = logits.argmax(dim=-1)
                accelerator.log({"batch/batch_step": idx, "batch/loss": loss, "batch/accuracy": accuracy.compute(predictions=predictions, references=batch["labels"])["accuracy"]})
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

    ender.record()
    torch.cuda.synchronize()
    training_time = starter.elapsed_time(ender)
    accelerator.log({"train": {"train_time": training_time}})

    # Saving model
    accelerator.wait_for_everyone()
    model = accelerator.unwrap_model(model)
    state_dict = model.state_dict()
    filename = data_folder + "models/BERT/model.pt"
    accelerator.save(state_dict, filename)

    accelerator.end_training()
    if accelerator.is_main_process:
        wandb.init(
            project="DMOZ-classification", 
            name="MODEL_" + str(wandb_id), 
            group=str(wandb_id),
            job_type="model",
            tags=["model"],
        )
        model_artifact = wandb.Artifact(
            name="model_" + MODEL_NAME.upper() + "_DMOZ", 
            type="model"
        )
        model_artifact.add_file(filename)
        wandb.log_artifact(model_artifact)

        wandb.finish()


if __name__ == "__main__":
    main()
