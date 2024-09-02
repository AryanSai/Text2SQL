import json

with open('./data/spider/dev_with_hardness.json', 'r') as file:
    og_dataset = json.load(file)

final_list=[]
final_list_filename = "input_list_SPIDER.json"

with open(final_list_filename, 'r') as file:
    data = json.load(file)
final_list = data.get('final_list', [])

# print(final_list)

extracted_data = [og_dataset[index] for index in final_list]

output_filename = "preprocessed_final_list.json"
with open(output_filename, 'w', encoding='utf-8') as output_file:
    json.dump(extracted_data, output_file, ensure_ascii=False, indent=4)

print(f"Extracted data has been saved to {output_filename}")
