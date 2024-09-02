import argparse, json
from collections import defaultdict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', dest='input', type=str, required=True)
    parser.add_argument('--original_dev', dest='original_dev', type=str, required=True)
    parser.add_argument('--output', dest='output', type=str, required=True)
    
    args = parser.parse_args()
    
    input_json=args.input
    original_dev =args.original_dev
    output =args.output
    
    with open(input_json, 'r') as f:
        data = json.load(f)

    grouped_data = defaultdict(lambda: {"input": "", "db_id": "", "sql": []})

    for entry in data:
        input_str = entry["input"]
        db_id = entry["db_id"]
        sql = entry["sql"]

        grouped_key = (input_str, db_id)
        grouped_data[grouped_key]["input"] = input_str
        grouped_data[grouped_key]["db_id"] = db_id
        grouped_data[grouped_key]["sql"].append(sql)

    result = [
        {
            "input": value["input"],
            "db_id": value["db_id"],
            "pred": value["sql"]
        }
        for value in grouped_data.values()
    ]

    with open(original_dev, 'r') as f:
        data2 = json.load(f)

    merged_data = []

    question_dict = {entry["question"]: entry for entry in data2}

    for entry in result:
        input_str = entry["input"].split('|')[0].strip()

        if input_str in question_dict:
            merged_entry = {
                "input": entry["input"],
                "db_id": entry["db_id"],
                "pred": entry["pred"],
                "gold": question_dict[input_str]["query"],
                "query_toks": question_dict[input_str]["query_toks"],
                "query_toks_no_value": question_dict[input_str]["query_toks_no_value"],
                "question": question_dict[input_str]["question"],
                "question_toks": question_dict[input_str]["question_toks"],
                "sql": question_dict[input_str]["sql"],
                "hardness": question_dict[input_str]["hardness"]
            }
            merged_data.append(merged_entry)

    with open(output, 'w') as f:
        json.dump(merged_data, f, indent=4)

    print("Preparation done for consistency!!")
