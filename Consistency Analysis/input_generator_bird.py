import argparse,json, random

def save_json(final_list, repeat, filename):
    data = {
        'final_list': final_list,
        'repeat': repeat
    }
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)  
    print(f"JSON saved to {filename}")

def load_json(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    final_list = data.get('final_list', [])
    repeat = data.get('repeat', 1)
    print(f"JSON loaded from {filename}")
    return final_list, repeat

def divide_samples(number_of_samples, simple_pct, moderate_pct, challenging_pct):
    assert simple_pct + moderate_pct + challenging_pct == 100, "Percentages must sum up to 100!"

    simple_samples = int((simple_pct / 100) * number_of_samples)
    moderate_samples = int((moderate_pct / 100) * number_of_samples)
    challenging_samples = number_of_samples - simple_samples - moderate_samples 

    return [simple_samples, moderate_samples, challenging_samples]

def pick_difficulty_lists(dataset, number_of_samples,repeat):
    with open(dataset, 'r') as f:
        data = json.load(f)
    
    easy = []
    medium = []
    hard = []

    for i, entry in enumerate(data):
        if entry.get("difficulty") == "simple":
            easy.append(i)
        elif entry.get("difficulty") == "moderate":
            medium.append(i)
        elif entry.get("difficulty") == "challenging":
            hard.append(i)

    print("og easy=",len(easy))
    print("og mediumm=",len(medium))
    print("og hard=",len(hard))
    
    # randomly sample indices from each category
    easy_sample = random.sample(easy, min(number_of_samples[0], len(easy)))
    medium_sample = random.sample(medium, min(number_of_samples[1], len(medium)))
    hard_sample = random.sample(hard, min(number_of_samples[2], len(hard)))
    print("easy=",len(easy_sample))
    print("mediumm=",len(medium_sample))
    print("hard=",len(hard_sample))
    repeated_easy = [item for item in easy_sample for _ in range(repeat)]
    repeated_medium = [item for item in medium_sample for _ in range(repeat)]
    repeated_hard = [item for item in hard_sample for _ in range(repeat)]

    combined_list = repeated_easy + repeated_medium + repeated_hard
    
    random.shuffle(combined_list)
    
    return combined_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', dest='dataset', type=str, required=True)
    parser.add_argument('--repeat', dest='repeat', type=int, default=7)
    parser.add_argument('--total', dest='total', type=int, default=50)
    parser.add_argument('--simple', dest='simple', type=int, default=20)
    parser.add_argument('--moderate', dest='moderate', type=int, default=30)
    parser.add_argument('--challenging', dest='challenging', type=int, default=50)
    parser.add_argument('--outputjson', dest='outputjson', type=str, required=True)
    
    args = parser.parse_args()
    
    dataset = args.dataset
    repeat = args.repeat
    total = args.total
    simple =args.simple
    moderate = args.moderate
    challenging = args.challenging
    outputjson = args.outputjson
    
    number_of_samples = divide_samples(total, simple, moderate, challenging)
    print("number of sample: ",number_of_samples)
    # number_of_samples =[1,1,1]
    
    final_list = pick_difficulty_lists(dataset, number_of_samples, repeat)
    
    save_json(final_list, repeat, outputjson)

    loaded_list = load_json('input_list_BIRD.json')
    
    print(loaded_list)
    print('---List Generation Complete---')