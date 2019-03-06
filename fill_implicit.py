import time, pandas as pd, numpy as np, math, re
from collections import defaultdict, Counter

users_list = pd.read_csv('data/corresponding_user.csv', header=None).values
books_list = pd.read_csv('data/corresponding_book.csv', delimiter='{', header=None, engine='python').values
# original_index = users_list[:, -1]
users_list = users_list[:, :2] # only new_index and userID
implicit_list = pd.read_csv('data/implicit_ratings.csv').values[:, 1:]
result = []

print(books_list)
# cache
last_seen_bookID = ('-math.inf', -math.inf) #(isbn, new_bookID)
last_seen_userID = ('-math.inf', -math.inf) #(userID, new_userID)

# implicit_list.shape[0]
for i in range(300000, 300000+100000):
	# init
	isbn = implicit_list[i, 0]
	userID = implicit_list[i, 1]
	# user part
	new_userID = math.inf
	if userID == last_seen_userID[0]:
		new_userID = last_seen_userID[1]
	else:
		for u in range(0, users_list.shape[0]):
			if userID == users_list[u, 1]:
				new_userID = users_list[u, 0]
				break
	# book part
	new_bookID = math.inf
	if isbn == last_seen_bookID[0]:
		new_bookID = last_seen_bookID[1]
	else:
		for b in range(0, books_list.shape[0]):
			if isbn == books_list[b, 1]:
				new_bookID = books_list[b, 0]
				break
	# record
	if not math.isinf(new_userID) and not math.isinf(new_bookID):
		last_seen_bookID = (isbn, new_bookID)
		last_seen_userID = (userID, new_userID)
		print('round', i, 'pair:', new_userID, new_bookID)
		result.append([new_userID, new_bookID])

# save
csv_columns=['new_userID', 'new_bookID']
pd.DataFrame(np.array(result), columns=csv_columns).to_csv('new_implicit.csv', sep=',', index=False)