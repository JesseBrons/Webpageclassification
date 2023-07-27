import re
import os
import sys
import nltk
import json
import wandb
import joblib
import datasets
import numpy as np
import pandas as pd
from time import process_time
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

np.random.seed(42)

class LemmaTokenizer:
        ignore_tokens = [',', '.', ';', ':', '"', '``', "''", '`']
        def __init__(self):
            self.wnl = WordNetLemmatizer()
        def __call__(self, doc):
            return [self.wnl.lemmatize(t) for t in word_tokenize(doc) if t not in self.ignore_tokens]

def prepare_dataset(data_folder, label2id, data_types, max_length):
    def combine_data(example):
        temp_text = ""
        for data_type in data_types:
            temp_text += example[data_type] + " "
        example["text"] = temp_text[:max_length]
        return example

    dataset = datasets.load_from_disk(data_folder + "dataset/")
    dataset = dataset["train"]
    dataset_encoded = dataset.class_encode_column("category")
    dataset_aligned = dataset_encoded.align_labels_with_mapping(label2id, "category")
    dataset_cleaned = dataset_aligned.map(combine_data)

    dataset = dataset_cleaned.remove_columns(["title", "body"])
    dataset = dataset.rename_column("category", "label")
    return dataset

def main():
    hps = {
        "data_types": ["title", "body"],
        "loss_function": "squared_hinge",
        "ngram_range": 3,
        "max_length": 512,
    }

    run = wandb.init(
        project="DMOZ-classification", 
        config=hps,
        job_type="training",
        name="SVM_DMOZ",
        tags=["SVM", "DMOZ"],
    )

    data_folder = "/ceph/csedu-scratch/other/jbrons/thesis-web-classification/"

    id2label = {0: "Arts", 1: "Business", 2: "Computers", 3: "Health", 4: "Home", 5: "News", 6: "Recreation", 7: "Reference", 8: "Science", 9: "Shopping", 10: "Society", 11: "Sports", 12: "Games"}
    label2id = {v: k for k, v in id2label.items()}
    labels = label2id.keys()

    dataset = prepare_dataset(data_folder, label2id, hps["data_types"], hps["max_length"])
    X_train, y_train = dataset["text"], dataset["label"]
    
    tokenizer=LemmaTokenizer()

    pipeline = make_pipeline(
        TfidfVectorizer(
            ngram_range=(1, hps["ngram_range"]),
            tokenizer=tokenizer,
            token_pattern=None
        ),
        LinearSVC(loss=hps["loss_function"])
    )

    t0 = process_time()
    pipeline.fit(X_train, y_train)
    training_time = process_time() - t0

    print("Training time {:5.2f}s for {:0d} samples.".format(training_time, len(y_train)))
    run.summary["training_time"] = training_time
    
    filename = data_folder + "models/SVM/model.pkl"
    joblib.dump(pipeline, filename, compress=3)

    model_artifact = wandb.Artifact(
        name="model_SVM_DMOZ",
        type="model"
    )
    model_artifact.add_file(thesis_folder + "models/SVM/model.pkl")
    run.log_artifact(model_artifact)

if __name__ == "__main__":
    main()
