import json

def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

if __name__ == "__main__":
    json_file_path = "input_list_BIRD.json"  
    data = load_json(json_file_path)
    total_components=len(data['final_list'])
    print(f"Total number of components in the JSON: {total_components}")