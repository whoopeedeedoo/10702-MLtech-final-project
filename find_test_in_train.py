import time, pandas as pd, numpy as np, math, re
from collections import defaultdict, Counter

# train part
train_list = []
with open('data/new_trainData.csv', 'r') as f:
	pass_header = False
	for line in f.readlines():
		if not pass_header:
			pass_header = True
			continue
		else:
			row = re.sub('\n', '', line).split(',')
			train_list.append(row[0]+','+row[1])
# test part
test_list = []
with open('data/new_testData.csv', 'r') as f:
	pass_header = False
	for line in f.readlines():
		if not pass_header:
			pass_header = True
			continue
		else:
			row = re.sub('\n', '', line).split(',')
			test_list.append(row[0]+','+row[1])

# check
for i in range(0, len(test_list)):
	# print('')
	test_data = test_list[i]
	if test_data in train_list:
		print(test_data)

print('program ends')