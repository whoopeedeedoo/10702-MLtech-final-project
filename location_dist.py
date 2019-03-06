# coding=utf-8
import time, pandas as pd, numpy as np, math, re, codecs
from collections import defaultdict, Counter

users_list = pd.read_csv('data/corresponding_user.csv', header=None).values
temp = np.zeros((users_list.shape[0], 1))
# ages, original_index
temp = np.concatenate([temp, temp], axis=1)
users_list = np.concatenate([users_list, temp], axis=1)
users_info = pd.read_csv('data/users.csv').values
users_ID = users_info[:, 0]

box = defaultdict(list)
count = 0
for u in users_info:
	print('round', count)
	locations = re.sub('[ \"]', '', u[1]).split(',')
	country = locations[-1]
	# too many usa
	# if country == 'usa':
	# 	country = locations[-2]
	if box[country]==[]:
		box[country] = 1
	else:
		box[country] += 1
	count+=1
print('write location_dist.csv')
c = Counter(box)
with codecs.open('location_dist.csv', 'w', "utf-8-sig") as f:
	f.write('country,distribution\n')
	for country, dist in c.most_common(1000):
		f.write(country+','+str(dist/count)+'\n')