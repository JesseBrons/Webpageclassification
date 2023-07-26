# Efficient Inference for Web Page Classification: Analysis of Methods
This is the code for the master thesis "Efficient Inference for Web Page Classification: Analysis of Methods".
This reposity provides the tools necessary to create our version of the DMOZ dataset. Furthermore, we provide the scripts to reproduce the experimental results in our thesis.

## Setup Environment

```bash
conda create -n web_classification python=?
conda activate web_classification
```

```bash
pip install -r requirements.txt
```

## Generate Dataset
First download the **content.rdf.u8** file from Curlie/DMOZ and place this file in the **dataset** folder. Run the crawl_websites script for each category you wish to retrieve.
```bash
bash create_index.sh
bash crawl_websites.sh
```

We will now truncate the files to the requested length and apply a first clean on the text. Run the truncate_data.sh for each category you wish to truncate and clean. Do not forget to change the output folders if necessary.
```bash
bash truncate_data.sh
```

For our last step we will 

## Run Experiments

##
