set -e
device="1"
REPEAT=7
# thresholds
EXACT=0.5
EXEC=0.5
CONSISTENCY=0.5
ITERATIONS=100

if [ $1 = "base" ]
then
    model_path="./models/text2sql-t5-base/checkpoint-39312"
    batch_size=4
elif [ $1 = "large" ]
then
    model_path="./models/text2sql-t5-large/checkpoint-30576"
    batch_size=1
elif [ $1 = "3b" ]
then
    model_path="./models/text2sql-t5-3b/checkpoint-103292"
    batch_size=1
else
    echo "The first arg must in [base, large, 3b]."
    exit
fi

if [ $2 = "spider" ]
then
    # spider's dev set
    table_path="./data/spider/tables.json"
    db_path="../Datasets/spider/database/"
    MODEL="RESDSQL_SPIDER"
    INPUT_LIST="../input_list_SPIDER.json"
    original_dev="./data/spider/dev_with_hardness.json"
    RESULTS_DIR="Results/Spider"
elif [ $2 = "bird" ]
then
    table_path="../Datasets/bird/dev_tables.json"
    db_path="../Datasets/bird/databases"
    MODEL="RESDSQL_BIRD"
    INPUT_LIST="../input_list_BIRD.json"
    original_dev="../Datasets/bird/dev.json"
    RESULTS_DIR="Results/Bird"
else
    echo "The second arg must in [spider, bird]."
    exit
fi

# python preprocess_final_list.py \
#     --original_dev $original_dev \
#     --input $INPUT_LIST\
#     --output "${RESULTS_DIR}/preprocessed_final_list_${MODEL}.json" \
    
# echo preprocess test set
# python preprocessing.py \
#     --mode "test" \
#     --table_path $table_path \
#     --input_dataset_path "${RESULTS_DIR}/preprocessed_final_list_${MODEL}.json" \
#     --output_dataset_path "${RESULTS_DIR}/preprocessed_dev_${MODEL}.json" \
#     --db_path $db_path \
#     --target_type "sql" 

# echo predict probability for each schema item
# python schema_item_classifier.py \
#     --batch_size 32 \
#     --device $device \
#     --seed 42 \
#     --save_path "./models/text2sql_schema_item_classifier" \
#     --dev_filepath "${RESULTS_DIR}/preprocessed_dev_${MODEL}.json" \
#     --output_filepath "${RESULTS_DIR}/test_with_probs_resdsql_${MODEL}.json" \
#     --use_contents \
#     --add_fk_info \
#     --mode "test"

# echo generate text2sql test set
# python text2sql_data_generator.py \
#     --input_dataset_path "${RESULTS_DIR}/test_with_probs_resdsql_${MODEL}.json" \
#     --output_dataset_path "${RESULTS_DIR}/resdsql_test_${MODEL}.json" \
#     --topk_table_num 4 \
#     --topk_column_num 5 \
#     --mode "test" \
#     --use_contents \
#     --add_fk_info \
#     --output_skeleton \
#     --target_type "sql"

# echo inference using the best text2sql ckpt
# python text2sql.py \
#     --batch_size $batch_size \
#     --device 1 \
#     --seed 42 \
#     --save_path $model_path \
#     --mode "eval" \
#     --dev_filepath "${RESULTS_DIR}/resdsql_test_${MODEL}.json" \
#     --original_dev_filepath $original_dev \
#     --db_path $db_path \
#     --num_beams 8 \
#     --num_return_sequences 8 \
#     --target_type "sql" \
#     --output "${RESULTS_DIR}/resdqsl_output.json"

# echo preparing for consistency
# python3 prepare_for_consistency.py\
#     --input "${RESULTS_DIR}/resdqsl_output.json"\
#     --original_dev $original_dev \
#     --output "${RESULTS_DIR}/resdqsl_output_for_consistency_${MODEL}.json"

echo consistency check
if [ $2 = "spider" ]
then
    CUDA_VISIBLE_DEVICES=0 python3 consistency_spider.py \
    --model $MODEL\
    --dbdir $db_path\
    --dbtable $table_path\
    --inputjson "${RESULTS_DIR}/resdqsl_output_for_consistency_${MODEL}.json"\
    --output "${RESULTS_DIR}/" \
    --exact_threshold $EXACT \
    --exec_threshold $EXEC \
    --consistency_threshold $CONSISTENCY \
    --repeat $REPEAT
    python3 ves_spider.py \
        --model $MODEL \
        --input_file "${RESULTS_DIR}/resdqsl_output_for_consistency_${MODEL}.json" \
        --output "$RESULTS_DIR/" \
        --iterations $ITERATIONS
elif [ $2 = "bird" ]
then
    CUDA_VISIBLE_DEVICES=0 python3 consistency_bird_new.py \
    --model $MODEL\
    --dbdir $db_path\
    --dbtable $table_path\
    --inputjson "${RESULTS_DIR}/resdqsl_output_for_consistency_${MODEL}.json"\
    --output "${RESULTS_DIR}/" \
    --exact_threshold $EXACT \
    --exec_threshold $EXEC \
    --consistency_threshold $CONSISTENCY \
    --repeat $REPEAT
    python3 ves_bird.py \
    --model $MODEL \
    --input_file "${RESULTS_DIR}/resdqsl_output_for_consistency_${MODEL}.json" \
    --output "${RESULTS_DIR}/" \
    --iterations $ITERATIONS
else
    echo "The second arg must in [spider, bird]."
    exit
fi