import os
import json
import numpy as np
import random


ROOT = "../../Dataset/"
TRAINING = "Training"
VALIDATION = "Validation"
TESTING = "Testing"
SUPPORT = [TRAINING, VALIDATION]

print(os.listdir(ROOT))

reference_dict = {}
support_dict = {}

for sub in os.listdir(ROOT):
	reference_dict[sub] = {}
	if sub in SUPPORT:
		support_dict[sub] = {}

print(reference_dict)
print(support_dict)

training_order_keys = []
for sub in list(iter(reference_dict)):
	for scene in os.listdir(ROOT + sub):
		reference_dict[sub][scene] = {}
		if sub in SUPPORT:
			support_dict[sub][scene] = {}
			if sub == TRAINING:
				training_order_keys.append(scene)
print(training_order_keys)

print(reference_dict)
print(support_dict)

max_common_size = 5000
for sub in list(iter(reference_dict)):
	for scene in os.listdir(ROOT + sub):
		max_common_size = min(
			max_common_size, len(os.listdir(ROOT + sub + '/' + scene + '/Colors'))
		)
print("max: ", max_common_size)

scale = 1.0
shift = 10
random_distance = 60
for sub in list(iter(reference_dict)):
	for scene in os.listdir(ROOT + sub):
		reference_dict[sub][scene] = []
		if sub in SUPPORT:
			support_dict[sub][scene] = []
		for num in range(1, int(max_common_size * scale), shift):
			reference_dict[sub][scene].append(num)
			if sub in SUPPORT:
				lb = num - random_distance if num - random_distance > 0 else 0
				ub = num + random_distance if num + random_distance < max_common_size \
					else max_common_size
				support_dict[sub][scene].append(np.random.randint(lb, ub))

print(reference_dict)
print(support_dict)

for epoch_step in range(max_common_size // shift + 1):
	print("EPOCH STEP: ", epoch_step)
	random.shuffle(training_order_keys)
	print("random shuffle: ", training_order_keys)
	for scene in training_order_keys:
		reference_dict[TRAINING][scene].pop()
		support_dict[TRAINING][scene].pop()
	print(reference_dict)
	print(support_dict)
	print()

write = False
if write:
	with open('reference.json', 'w') as fp:
		json.dump(reference_dict, fp, sort_keys=True, indent=2)

	with open('support.json', 'w') as fp:
		json.dump(support_dict, fp, sort_keys=True, indent=2)
