# set_A = {1, 2, 3, 4, 5}
# set_B = {1,1,2,3,4,5}


# print(set_A.symmetric_difference(set_B))
# print(set_A==set_B)


predicted_set = [
    ("Laptop", 2, 1000),
    ("Mouse", 5, 50),
    ("Laptop", 2, 1000)  # Duplicate row
]

ground_truth_set = [
    ("Laptop", 2, 1000),
    ("Mouse", 5, 50)
]

print('----------------------------------------------------------')
print(predicted_set)
print('----------------------------------------------------------')

print('----------------------------------------------------------')
print(ground_truth_set)
print('----------------------------------------------------------')

print(predicted_set == ground_truth_set)  # Output: True (Incorrect!)

# print(predicted_set.symmetric_difference(ground_truth_set))
# print(set_A==set_B)

