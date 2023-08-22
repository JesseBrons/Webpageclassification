import re
import os
import sys
import nltk
import json
import wandb
import joblib
import datasets
from time import process_time
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

MODEL_VER = str(sys.argv[1])

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
    dataset = dataset["test"]
    dataset_encoded = dataset.class_encode_column("category")
    dataset_aligned = dataset_encoded.align_labels_with_mapping(label2id, "category")
    dataset_cleaned = dataset_aligned.map(combine_data)

    dataset = dataset_cleaned.remove_columns(["title", "body"])
    dataset = dataset.rename_column("category", "label")
    return dataset

def main():
    hps = {
        "data_types": ["title", "body"],
        "max_length": 512,
    }

    run = wandb.init(
        project="DMOZ-classification",
        name="SVM_DMOZ_INFERENCE",
        config=hps,
        job_type="inference",
        tags=["SVM", "DMOZ"]
    )

    data_folder = str(sys.argv[2])
    
    id2label = {0: "Arts", 1: "Business", 2: "Computers", 3: "Health", 4: "Home", 5: "News", 6: "Recreation", 7: "Reference", 8: "Science", 9: "Shopping", 10: "Society", 11: "Sports", 12: "Games"}
    label2id = {v: k for k, v in id2label.items()}
    labels = label2id.keys()
    
    dataset = prepare_dataset(data_folder, label2id, hps["data_types"], hps["max_length"])
    X_test, y_test = dataset["text"], dataset["label"]
    
    model_artifact = run.use_artifact("model_SVM_DMOZ:" + MODEL_VER)
    model_dir = model_artifact.download(data_folder + "artifacts/")
    loaded_model = joblib.load(model_dir + "/model.pkl")

    t0 = process_time()
    predictions = loaded_model.predict(X_test)
    inference_time = process_time() - t0

    wandb.summary["inference_time"] = inference_time * 1000 
    wandb.summary["accuracy"] = accuracy_score(y_test, predictions)
    wandb.summary["f1_weighted"] = f1_score(y_test, predictions, average="weighted")
    wandb.summary["precision"] = precision_score(y_test, predictions, average="weighted")
    wandb.summary["recall"] = recall_score(y_test, predictions, average="weighted")

if __name__ == "__main__":
    main()
