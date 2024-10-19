#!/bin/bash

INPUT_FILE="Results/Bird/codestral/Codestral-22B-v0.1-Q8_0_Consistency_Results_BIRD.csv"

ITERATIONS=100

echo First Run Evaluation------------------------------------------------
python3 firstrun_bird.py \
    --input_file $INPUT_FILE 

echo VES Evaluation------------------------------------------------
python3 ves_bird_first.py \
    --input_file $INPUT_FILE \
    --iterations $ITERATIONS