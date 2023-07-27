mixed_precision_list=(no fp16)
model_list=(bert distilbert tinybert)

NUMBER_GPU=$1
MIXED_PREC=$2
MODEL_NAME=$3

if [[ ! " ${mixed_precision_list[*]} " =~ " ${MIXED_PREC} " ]]; then
    echo "Invalid argument! Mixed precision ${MIXED_PREC} is not in (${mixed_precision_list[*]})."
    exit
fi

if [[ ! " ${model_list[*]} " =~ " ${MODEL_NAME} " ]]; then
    echo "Invalid argument! Model ${MODEL_NAME} is not in (${model_list[*]})."
    exit
fi

MAIN_DIR=.

accelerate launch --num_processes=$NUMBER_GPU --num_machines=1 --mixed_precision="no" --dynamo_backend="no" ./training/DMOZ_BERT.py $MODEL_NAME "$MAIN_DIR"
