import time, pandas as pd, numpy as np, math, re
from collections import defaultdict, Counter

users_list = pd.read_csv('data/corresponding_user.csv', header=None).values
original_index = users_list[:, -1]
users_list = users_list[:, :2] # only new_index and userID
temp = np.zeros((users_list.shape[0], 1))
# ages, location
temp = np.concatenate([temp, temp], axis=1)
users_list = np.concatenate([users_list, temp, original_index.reshape(-1, 1)], axis=1)
users_info = pd.read_csv('data/users.csv').values
users_ID = users_info[:, 0]
# ages part
ages = users_info[:, 2]
avg_age = 0 #34.75143370454978,  no children no elders 36.60317973201097
# location part
location_list = pd.read_csv('data/location_dist.csv').values
location_threshold = 0.04
should_be_removed = np.where(location_list[:, 1]<=location_threshold)[0]
location_list = np.delete(location_list, should_be_removed, axis=0)
# remove 'nan' country
for i in range(0, location_list.shape[0]):
	if type(location_list[i, 0]) != str and np.isnan(location_list[i, 0]):
		location_list = np.delete(location_list, i, axis=0)
		i-=1
		break
location_list = location_list[:, 0]
# evaluate avg age
counter = 0
childrean_threshold = 8
elder_threshold = 88
children_table = defaultdict(list)
elder_table = defaultdict(list)
# for i in range(0, 18+1):
# 	children_table[str(i)] = 0
for i in range(0, ages.shape[0]):
	if np.isnan(ages[i]):
		continue
	else:
		if ages[i]<=childrean_threshold:
			if children_table[(int(ages[i]))] == []:
				children_table[(int(ages[i]))] = 1
			else:
				children_table[(int(ages[i]))] +=1
			ages[i] = 35 #childrean_threshold #34.78687638830522
		elif ages[i]>=elder_threshold:
			if elder_table[(int(ages[i]))] == []:
				elder_table[(int(ages[i]))] = 1
			else:
				elder_table[(int(ages[i]))] +=1
			ages[i] = 35 #elder_threshold#34.78687638830522
		avg_age+=ages[i]
		counter+=1
avg_age /= counter 
print('avg_age:', avg_age)
children_table = Counter(children_table)
elder_table = Counter(elder_table)
for age in range(0, childrean_threshold+1):
	if children_table[age]>0:
		print(age, children_table[age])
for age in range(elder_threshold, elder_threshold+1000):
	if elder_table[age]>0:
		print(age, elder_table[age])
# exit()
# fill avg age
# avg_age =  35#34.78687638830522
for i in range(0, ages.shape[0]):
	if np.isnan(ages[i]):
		ages[i]  = 35
# fill correspoding info
# 
for i in range(0, users_list.shape[0]):
	print('round', i)
	ordered_user = users_list[i, 1]
	idx_in_info = math.inf
	# find correspoding ID
	# for j in range(0, users_ID.shape[0]):
	# 	if ordered_user == users_ID[j]:
	# 		idx_in_info = j
	# 		break
	idx_in_info = original_index[i]
	# age
	users_list[i, 2] = str(ages[idx_in_info])
	# location
	location = re.sub('[ \"]', '', users_info[idx_in_info, 1]).split(',')[-1]
	location = np.where(location_list==location)[0]
	if location.shape[0] == 0:
		# not listed
		# other country
		location = location_list.shape[0]
	else:
		location = location[0]
	users_list[i, 3] = str(location)
	# original index in users.csv
	users_list[i, -1] = str(int(idx_in_info))
	# remove row, doesnot speed up
	# users_ID = np.delete(users_ID, [idx_in_info], axis=0)
	# ages = np.delete(ages, [idx_in_info], axis=0)

# print(users_list.shape)
# np.savetxt('corresponding_user_info.csv', users_list, delimiter=",", fmt='%s,%s'+',%s'*temp.shape[1] + ',%s')
csv_columns=['new_userID', 'userID', 'age', 'location', 'users_csv_index']
pd.DataFrame(users_list, columns=csv_columns).to_csv('data/corresponding_user_info.csv', sep=',', index=False)
# with open('corresponding_user_info.csv', 'w') as f:
# 	csv_columns=['new_userID', 'userID', 'age', 'location', 'users_csv_index']
# 	f.write(','.join(csv_columns)+'\n')
# 	for u in users_list:
# 		f.write(str(u[0]))
# 		if u.shape[0]>1:
# 			for i in range(1, u.shape[0]):
# 				f.write(','+str(u[i]))
# 		f.write('\n')