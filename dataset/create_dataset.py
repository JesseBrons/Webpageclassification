import json 
import sys
import os
import numpy as np
import datasets

target_folder = sys.argv[1]
data_folder = sys.argv[2]

categories = ["Arts", "Business", "Computers", "Games", "Home", "Health", "News", "Recreation", "Reference", "Science", "Shopping", "Society", "Sports"] #Home
features = ["category", "title", "body", "keywords"]

for idx, category in enumerate(categories):
    data_path = data_folder + "/" + category + "/*.txt"
    dataset_files = {category: data_path}
    dataset_temp = datasets.load_dataset("json", split=category, data_files=dataset_files, num_proc=16)
    dataset_temp = dataset_temp.remove_columns([col for col in dataset_temp.column_names if col not in features])
    
    if idx == 0:
        dataset = dataset_temp
    else:
        dataset = datasets.concatenate_datasets([dataset, dataset_temp])

dataset = dataset.train_test_split(test_size=0.2, shuffle=True)

dataset.save_to_disk(target_folder)
