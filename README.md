# Efficient Inference for Web Page Classification: Analysis of Methods
This is the code for the master thesis "Efficient Inference for Web Page Classification: Analysis of Methods".
This reposity provides the tools necessary to create our version of the DMOZ dataset. Furthermore, we provide the scripts to reproduce the experimental results in our thesis.

## Setup Environment

```bash
conda create -n web_classification python=3.10.6
conda activate web_classification
```

```bash
pip install -r requirements.txt
```

## Generate Dataset
Available categories for the dataset: _Arts, Business, Computers, Games, Health, Home, News, Recreation, Reference, Science, Shopping, Society and Sports_.

First download the **content.rdf.u8** file from Curlie/DMOZ and place this file in the **dataset** folder. Run the crawl_websites script for each category you wish to retrieve.
```bash
bash create_index.sh
bash crawl_websites.sh [CATEGORY]
```

We will now truncate the files to the requested length and apply a first clean on the text. Run the truncate_data.sh for each category you wish to truncate and clean. Do not forget to change the input/output folders if necessary.
```bash
bash truncate_data.sh [CATEGORY]
```

For our last step we need to create a Hugging Face dataset structure. Changes to the Python script can be made to use a different train test split and to only retrieve a specific set of categories and/or features. Do not forget to change the input/output folders if necessary. 
```bash
bash create_dataset.sh
```

## Weights and Biases
For our experiments we made use of Weights and Biases (WandB) to save and load models and to keep track of experiments. Check the Python scripts to make sure the right parameters are set for WandB. Note that in order to make use of WandB, you need an account and have placed your API key in the environment variables (WANDB_API_KEY).

## Train models
To train the SVM model run the code below.
```bash
bash train_model_SVM.sh
```

To train the BERT model run the code below. GPUS is equal to the number of GPUs you want to use. MIXED PRECISION can either be "no" for FP32 or "fp16" for FP16. MODEL can either be "bert", "distilbert" or "tinybert".
```bash
bash train_model_BERT.sh [GPUS] [MIXED PRECISION] [MODEL]
```

## Run Experiments
For these experiments you can use the latest model available on WandB or specify a model version. For each model you can specify the model parameters and add the optimization techniques in the "hps" variable inside the Python script.

To run inference on the SVM model run the code below. MODEL VERSION can either be "latest" or a specific version "v$" where dollar represents the version number.
```bash
bash inference_model_SVM.sh [MODEL VERSION]
```

To run inference on the BERT model run the code below. GPUS is equal to the number of GPUs you want to use. MIXED PRECISION can either be "no" for FP32 or "fp16" for FP16. MODEL can either be "bert", "distilbert" or "tinybert". MODEL VERSION can either be "latest" or a specific version "v$" where dollar represents the version number.
```bash
bash inference_model_BERT.sh [GPUS] [MIXED PRECISION] [MODEL] [MODEL VERSION]
```


