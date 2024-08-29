#!/bin/bash

DATASET=${1:-"spider"}  # default is "spider" 
REPEAT=5
MODEL_PATH="Models/codegemma-7b-Q8_0.gguf"
RESULTS_DIR="Results/"
TOTAL=50
SIMPLE=20
MODERATE=30
CHALLENGING=50

if [ "$DATASET" == "spider" ]; then
    DB_DIR="Datasets/spider/database"
    DB_TABLE="Datasets/spider/tables.json"
    DATASET_TAG="_SPIDER"
    INPUT_JSON="input_list_SPIDER.json"
    DATASET_FILE="Datasets/spider/dev_with_difficulty.json"
elif [ "$DATASET" == "bird" ]; then
    DB_DIR="Datasets/bird/databases"
    DB_TABLE="Datasets/bird/dev_tables.json"
    DATASET_TAG="_BIRD"
    INPUT_JSON="input_list_BIRD.json"
    DATASET_FILE="Datasets/bird/bird_dev_as_spider.json"
else
    echo "Unknown dataset: $DATASET. Please choose 'spider' or 'bird'."
    exit 1
fi

# python3 input_generator.py \
#     --dataset $DATASET_FILE \
#     --dataset_tag $DATASET_TAG \
#     --repeat $REPEAT\
#     --total $TOTAL \
#     --simple $SIMPLE \
#     --moderate $MODERATE \
#     --challenging $CHALLENGING 

CUDA_VISIBLE_DEVICES=0 python3 consistency.py \
    --modelpath $MODEL_PATH\
    --dataset $DATASET_FILE \
    --dbdir $DB_DIR\
    --dbtable $DB_TABLE\
    --dataset_tag $DATASET_TAG\
    --inputjson $INPUT_JSON \
    --output $RESULTS_DIR

# INPUT_FILE="Results/codegemma-7b-Q8_0_Consistency_Results_SPIDER.csv"
# ITERATIONS=20

# python3 ves.py \
#     --input_file $INPUT_FILE \
#     --iterations $ITERATIONS