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

def divide_samples(number_of_samples, easy_pct, medium_pct, hard_pct, extra_hard_pct):
    assert easy_pct + medium_pct + hard_pct + extra_hard_pct == 100, "Percentages must sum up to 100!"

    easy_samples = int((easy_pct / 100) * number_of_samples)
    medium_samples = int((medium_pct / 100) * number_of_samples)
    hard_samples = int((hard_pct / 100) * number_of_samples)
    extra_hard_samples = number_of_samples - easy_samples - medium_samples - hard_samples

    return [easy_samples, medium_samples, hard_samples, extra_hard_samples]

def pick_hardness_lists(dataset, number_of_samples,repeat):
    with open(dataset, 'r') as f:
        data = json.load(f)
    
    easy = []
    medium = []
    hard = []
    extra_hard = []

    for i, entry in enumerate(data):
        if entry.get("hardness") == "easy":
            easy.append(i)
        elif entry.get("hardness") == "medium":
            medium.append(i)
        elif entry.get("hardness") == "hard":
            hard.append(i)
        elif entry.get("hardness") == "extra":
            extra_hard.append(i)

    # randomly sample indices from each category
    easy_sample = random.sample(easy, min(number_of_samples[0], len(easy)))
    medium_sample = random.sample(medium, min(number_of_samples[1], len(medium)))
    hard_sample = random.sample(hard, min(number_of_samples[2], len(hard)))
    extra_hard_sample = random.sample(extra_hard, min(number_of_samples[3], len(extra_hard)))
    
    
    repeated_easy = [item for item in easy_sample for _ in range(repeat)]
    repeated_medium = [item for item in medium_sample for _ in range(repeat)]
    repeated_hard = [item for item in hard_sample for _ in range(repeat)]
    repeated_extra_hard = [item for item in extra_hard_sample for _ in range(repeat)]
    
    
    combined_list = repeated_easy + repeated_medium + repeated_hard + repeated_extra_hard
    
    random.shuffle(combined_list)
    
    # print(combined_list)
    
    return combined_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', dest='dataset', type=str, required=True)
    parser.add_argument('--repeat', dest='repeat', type=int, default=7)
    parser.add_argument('--total', dest='total', type=int, default=50)
    parser.add_argument('--easy', dest='easy', type=int, default=10)
    parser.add_argument('--medium', dest='medium', type=int, default=20)
    parser.add_argument('--hard', dest='hard', type=int, default=30)
    parser.add_argument('--extra', dest='extra', type=int, default=40)
    parser.add_argument('--output', dest='output', type=str, required=True)
    
    args = parser.parse_args()
    
    dataset = args.dataset
    repeat = args.repeat
    total = args.total
    easy =args.easy
    medium = args.medium
    hard = args.hard
    extra_hard = args.extra
    output = args.output
    
    number_of_samples = divide_samples(total, easy, medium, hard, extra_hard)
    # number_of_samples =[1,1,1]
    
    final_list = pick_hardness_lists(dataset, number_of_samples, repeat)
    
    save_json(final_list, repeat, output)

    loaded_list = load_json(output)
    
    # print(loaded_list)
    
    print('---List Generation Complete---')