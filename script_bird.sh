#!/bin/bash

REPEAT=7
MODEL_PATH="Models/Yi-Coder-9B-Chat-Q8_0.gguf"
RESULTS_DIR="Results/Bird/"
TOTAL=250
SIMPLE=20
MODERATE=30
CHALLENGING=50

#results are these 
# og easy= 925
# og mediumm= 464
# og hard= 145
# easy= 50
# mediumm= 75
# hard= 125

# thresholds
EXEC=0.5
CONSISTENCY=0.5

DB_DIR="Datasets/bird/databases"
DB_TABLE="Datasets/bird/dev_tables.json"
INPUT_JSON="input_list_BIRD.json"
DATASET_FILE="Datasets/bird/dev.json"

# echo Input Generation------------------------------------------------
# python3 input_generator_bird.py \
#     --dataset $DATASET_FILE \
#     --repeat $REPEAT\
#     --total $TOTAL \
#     --simple $SIMPLE \
#     --moderate $MODERATE \
#     --challenging $CHALLENGING \
#     --outputjson $INPUT_JSON

# echo Consistency Evaluation------------------------------------------------
# CUDA_VISIBLE_DEVICES=1 python3 consistency_bird.py \
#     --modelpath $MODEL_PATH\
#     --dataset $DATASET_FILE \
#     --dbdir $DB_DIR\
#     --dbtable $DB_TABLE\
#     --inputjson $INPUT_JSON \
#     --output $RESULTS_DIR \
#     --exec_threshold $EXEC \
#     --consistency_threshold $CONSISTENCY 

INPUT_FILE="Results/Bird/Yi-Coder-9B-Chat-Q8_0_Consistency_Results_BIRD.csv"
ITERATIONS=100

echo VES Evaluation------------------------------------------------
python3 ves_bird.py \
    --input_file $INPUT_FILE \
    --iterations $ITERATIONS