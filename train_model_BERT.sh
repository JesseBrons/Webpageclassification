NUMBER_GPU=$1
MIXED_PREC=$2
MODEL_NAME=$3

MAIN_DIR=.

accelerate launch --num_processes=$NUMBER_GPU --num_machines=1 --mixed_precision="no" --dynamo_backend="no" ./training/DMOZ_BERT.py $MODEL_NAME "$MAIN_DIR"