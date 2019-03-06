import time, pandas as pd, numpy as np, math, re, codecs
from collections import defaultdict, Counter

# ISBN,Book-Title,Book-Author,Year-Of-Publication,Publisher,Image-URL-S,Image-URL-M,Image-URL-L,Book-Description
books_info = pd.read_csv('data/books.csv').values
books_list = pd.read_csv('data/corresponding_book.csv', delimiter='{', header=None, engine='python').values
books_original_idx = pd.read_csv('data/books_original_idx.csv').values
books_year = books_info[:, 3]
year_box = []
avg_year = 1993.6895973909131; count = 0
# for i in range(0, books_year.shape[0]):
# 	year = books_year[i]
# 	if year > 0:
# 		# year_box.append(year)
# 		# print(year)	
# 		avg_year += year
# 		count += 1 
# print('avg_year:', avg_year/count)
# original_idx = []
# promblem_idx = []
years = np.zeros((books_list.shape[0], 1))
# publishers = np.zeros((books_list.shape[0], 1))
# temp = np.concatenate([years, publishers], axis=1)
books_list = np.concatenate([books_list, years, books_original_idx], axis=1)
# publisher_box = defaultdict(list)
# count = 0
publisher_threshold = 0.005
publisher_list = pd.read_csv('data/publisher_dist.csv').values
# print(np.sum(publisher_list[:500, 1]))
# exit()
should_be_removed = np.where(publisher_list[:, 1]<=publisher_threshold)[0]
publisher_list = np.delete(publisher_list, should_be_removed, axis=0)[:, 0]
for i in range(0, books_list.shape[0]): # books_list.shape[0]
	print('round: '+ str(i)+'/'+str(books_list.shape[0]))
	# year
	if books_original_idx[i]==-1:
		books_list[i, 2] = avg_year
	else:
		books_list[i, 2] = books_year[books_original_idx[i]][0]
	# publisher part
	# fill the publisher number
	# publisher = math.inf
	# if books_original_idx[i]==-1:
	# 	# other publisher
	# 	# books_list[i, 3] = str(publisher_list.shape[0])
	# 	# continue
	# 	publisher = publisher_list.shape[0]
	# else:
	# 	publisher = books_info[books_original_idx[i], 4][0]
	# 	if type(publisher) != str and math.isnan(publisher):
	# 		# other publisher
	# 		# books_list[i, 3] = str(publisher_list.shape[0])
	# 		# continue
	# 		publisher = publisher_list.shape[0]
	# 	else:
	# 		publisher = re.sub('[^a-zA-Z]', '', publisher).lower()
	# 		publisher = np.where(publisher_list==publisher)[0]
	# 		if publisher.shape[0] == 0:
	# 			# not listed
	# 			# other publisher
	# 			publisher = publisher_list.shape[0]
	# 		else:
	# 			publisher = publisher[0]
	# books_list[i, 3] = str(publisher)
	# find the dist
	# if books_original_idx[i]==-1:
	# 	pass
	# else:
	# 	publisher = books_info[books_original_idx[i], 4][0]
	# 	if type(publisher) != str and math.isnan(publisher):
	# 		publisher = ''
	# 	publisher = re.sub('[^a-zA-Z]', '', publisher).lower()
	# 	if publisher_box[publisher]==[]:
	# 		publisher_box[publisher] = 1
	# 	else:
	# 		publisher_box[publisher] += 1
	# 	count+=1

# report the publisher dist
# publisher_c = Counter(publisher_box)
# with codecs.open('publisher_dist.csv', 'w', "utf-8-sig") as f:
# 	f.write('publisher,distribution\n')
# 	for publisher, dist in publisher_c.most_common(1000):
# 		f.write(publisher+','+str(dist/count)+'\n')

# save book info
csv_columns = ['new_bookID', 'ISBN', 'year', 'original_idx']
pd.DataFrame(books_list, columns=csv_columns).to_csv('corresponding_book_info.csv', sep=',', index=False)

# try:
# 	start = 40000
# 	shift = 20000
# 	for i in range(start, start+shift): # books_list.shape[0]
# 		print('round: '+ str(i)+'/'+str(start+shift))
# 		idx_in_info = math.inf
# 		for j in range(0, books_info.shape[0]):
# 			if books_list[i, 1] == books_info[j, 0]:
# 				idx_in_info = j
# 		if math.isinf(idx_in_info):
# 			original_idx.append(-1)
# 		else:
# 			original_idx.append(idx_in_info)
# except Exception as e:
# 	pass
# finally:
# 	original_idx = np.array(original_idx)
# 	pd.DataFrame(original_idx, columns=['original_idx']).to_csv('books_original_idx_'+str(start+shift)+'.csv', sep=',', index=False)