import argparse,json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', dest='input', type=str, required=True)
    parser.add_argument('--original_dev', dest='original_dev', type=str, required=True)
    parser.add_argument('--output', dest='output', type=str, required=True)
    
    args = parser.parse_args()
    
    input_json=args.input
    original_dev =args.original_dev
    output =args.output
    
    with open(original_dev, 'r') as file:
        og_dataset = json.load(file)
    
    final_list=[]
    
    with open(input_json, 'r') as file:
        data = json.load(file)
    final_list = data.get('final_list', [])
    
    # print(final_list)
    
    extracted_data = [og_dataset[index] for index in final_list]
    
    with open(output, 'w', encoding='utf-8') as output_file:
        json.dump(extracted_data, output_file, ensure_ascii=False, indent=4)
    
    print(f"Extracted data has been saved to {output}")
