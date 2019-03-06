import time, pandas as pd, numpy as np, math, re, codecs, time, os, codecs
# from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict, Counter
import nltk
from nltk.corpus import stopwords
from sklearn import feature_extraction 
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

def tokenize_and_stem(text):
	# first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
	tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
	filtered_tokens = []
	# filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
	for token in tokens:
		if re.search('[a-zA-Z]', token):
			filtered_tokens.append(token)
	stems = [stemmer.stem(t) for t in filtered_tokens]
	return stems


def tokenize_only(text):
	# first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
	tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
	filtered_tokens = []
	# filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
	for token in tokens:
		if re.search('[a-zA-Z]', token):
			filtered_tokens.append(token)
	return filtered_tokens

# nltk.download()

time_start = time.time()
# ISBN,Book-Title,Book-Author,Year-Of-Publication,Publisher,Image-URL-S,Image-URL-M,Image-URL-L,Book-Description
books_disc = pd.read_csv('data/books.csv').values[:, -1]
N = books_disc.shape[0]
books_disc = books_disc[:N]
for i in range(0, books_disc.shape[0]):
	if type(books_disc[i]) != str:
		books_disc[i] = ''
	book = books_disc[i].lower()
	book = re.sub('[^a-z ]', '', book)
	book = re.sub(' {2}', ' ', book)
	books_disc[i] = book

stopwords = nltk.corpus.stopwords.words('english')+['']

stemmer = SnowballStemmer("english")


# totalvocab_stemmed = []
# totalvocab_tokenized = []
# count = 0
# for i in books_disc:
# 	print(chr(8)*50 + 'tokenize and stem round: '+str(count+1)+'/'+str(books_disc.shape[0]), end='')
# 	allwords_stemmed = tokenize_and_stem(i) #for each item in 'synopses', tokenize/stem
# 	totalvocab_stemmed.extend(allwords_stemmed) #extend the 'totalvocab_stemmed' list
    
# 	allwords_tokenized = tokenize_only(i)
# 	totalvocab_tokenized.extend(allwords_tokenized)
# 	count+=1
# print('')
# vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_stemmed)
# print('there are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame')
# print(vocab_frame.head())

print('tfidf_vectorizer')
tfidf_vectorizer = TfidfVectorizer(max_df=0.96, max_features=None,
									min_df=0.07, stop_words='english',
									use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))
tfidf_time = time.time()
tfidf_matrix = tfidf_vectorizer.fit_transform(books_disc) #fit the vectorizer to synopses
tfidf_time = time.time() - tfidf_time
print('tfidf_vectorizer time:', tfidf_time, 's')
# print(tfidf_matrix)
print(tfidf_matrix.shape)
terms = tfidf_vectorizer.get_feature_names()
print(terms)

print('KMeans')
num_clusters = 7

km = KMeans(n_clusters=num_clusters)
km_time = time.time()
km.fit(tfidf_matrix)
km_time = time.time() - km_time
print('KMeans time:', km_time, 's')
clusters = km.labels_.tolist()
# print(clusters)

print('distribute and save')
books_info = pd.read_csv('data/corresponding_book_info.csv').values[:N]
original_idx = books_info[:, -1].reshape(-1, 1).astype(int)
books_info = books_info[:, :-1]
# books_cate = np.array(clusters).reshape(-1, 1)
clusters = np.array(clusters)
corresponding_cate = []# books_cate[original_idx]
for i in range(0, books_info.shape[0]):
	cate = -1
	if original_idx[i] == -1:
		# other category
		cate = num_clusters
	else:
		cate = clusters[original_idx[i]][0]
	corresponding_cate.append(cate)
corresponding_cate = np.array(corresponding_cate).reshape(-1, 1)
books_info = np.concatenate([books_info, corresponding_cate, original_idx], axis=1)
csv_columns = ['new_bookID', 'ISBN', 'year', 'category', 'original_idx']
# pd.DataFrame(books_info, columns=csv_columns).to_csv('corresponding_book_info_with_category.csv', sep=',', index=False)

# N_books = 2000#books_disc.shape[0]
# TF_table = defaultdict(list)
# IDF_table = defaultdict(list)
# # term_table
# # nltk.download()

# stopwords_list = stopwords.words('english') + ['']
# stopwords = defaultdict(list)
# for stopword in stopwords_list:
# 	stopwords[stopword] = True
# N_terms = 10000
# N_words = 0
# N_kinds_words = 0
# N_valid_book = 0
# error_count = 0
# # collect terms
# for i in range(0, N_books):
# 	print(chr(8)*50 + 'collect TFIDF round: '+str(i+1)+'/'+str(N_books), end='')
# 	if type(books_disc[i]) != str:
# 		continue
# 	N_valid_book += 1 # 會不準
# 	book = books_disc[i].lower()
# 	book = re.sub('[^a-z ]', '', book).split(' ')
# 	for word in book:
# 		if stopwords[word] != [] or len(word)==1:
# 			continue
# 		# if word[-1] == 's':
# 		# 	# 解決大部分複數
# 		# 	word = word[:-1]
# 		# elif word[-2:] == 'ed':
# 		# 	# 解決大部分過去式
# 		# 	word = word[:-2]
# 		# elif word[-2:] == 'ly':
# 		# 	# 解決大部分副詞
# 		# 	word = word[:-2]
# 		# elif word[-3:] == 'ing':
# 		# 	# 解決大部分進行式
# 		# 	word = word[:-3]
# 		# if word == '':
# 		# 	continue
# 		# 收錄TF
# 		if N_kinds_words < N_terms:
# 			if TF_table[word] == []:
# 				TF_table[word] = defaultdict(list)
# 				TF_table[word][i] = 1
# 				N_kinds_words += 1
# 			elif TF_table[word][i] == []:
# 				TF_table[word][i] = 1
# 			else:
# 				TF_table[word][i] += 1
# 			# 收錄IDF
# 			if i not in IDF_table[word]:
# 				IDF_table[word].append(i)
# 		else:
# 			if TF_table[word] != []:
# 				if TF_table[word][i] == []:
# 					TF_table[word][i] = 1
# 				else:
# 					TF_table[word][i] += 1
# 			if i not in IDF_table[word] and IDF_table[word] != []:
# 				IDF_table[word].append(i)
# 		N_words += 1
# print('')
# # TF_table_c = Counter(TF_table)
# # print(TF_table_c.most_common(50))
# # print(IDF_table)
# # print(num_words)

# # 計算tf-idf
# tfidf_matrix = []
# for i in range(0, N_books):
# 	print(chr(8)*50 + 'compute TFIDF round: '+str(i+1)+'/'+str(N_books), end='')
# 	if type(books_disc[i]) != str:
# 		tfidf_matrix.append([0]*N_kinds_words)
# 		continue
# 	# print('book', i)
# 	tfidf_vec = []
# 	# TFIDF_table = defaultdict(list)
# 	book = books_disc[i].lower()
# 	book = re.sub('[^a-z ]', '', book).split(' ') 
# 	num_words_this_book = len(book)
# 	for word in TF_table:
# 		# print(word)
# 		# print(TF_table[word])
# 		if TF_table[word]==[]:
# 			continue
# 		if TF_table[word][i] == []:
# 			tfidf_vec.append(0)
# 			continue
# 		tf = TF_table[word][i]/num_words_this_book
# 		idf = math.log(N_valid_book/len(IDF_table[word]))
# 		tfidf_vec.append(tf*idf)
# 	tfidf_matrix.append(tfidf_vec)
# print('')
# tfidf_matrix = np.array(tfidf_matrix)
# # print(tfidf_matrix)
# csv_columns = ['new_bookID', 'ISBN', 'year', 'original_idx']
# pd.DataFrame(tfidf_matrix, columns=None).to_csv('book_tfidf.csv', sep=',', index=False)





# books_list = pd.read_csv('data/corresponding_book.csv', delimiter='{', header=None, engine='python').values
# books_original_idx = pd.read_csv('data/books_original_idx.csv').values
# books_year = books_info[:, 3]
# year_box = []
# avg_year = 1993.6895973909131; count = 0
# # for i in range(0, books_year.shape[0]):
# # 	year = books_year[i]
# # 	if year > 0:
# # 		# year_box.append(year)
# # 		# print(year)	
# # 		avg_year += year
# # 		count += 1 
# # print('avg_year:', avg_year/count)
# # original_idx = []
# # promblem_idx = []
# years = np.zeros((books_list.shape[0], 1))
# # publishers = np.zeros((books_list.shape[0], 1))
# # temp = np.concatenate([years, publishers], axis=1)
# books_list = np.concatenate([books_list, years, books_original_idx], axis=1)
# # publisher_box = defaultdict(list)
# # count = 0
# publisher_threshold = 0.005
# publisher_list = pd.read_csv('data/publisher_dist.csv').values
# # print(np.sum(publisher_list[:500, 1]))
# # exit()
# should_be_removed = np.where(publisher_list[:, 1]<=publisher_threshold)[0]
# publisher_list = np.delete(publisher_list, should_be_removed, axis=0)[:, 0]
# for i in range(0, books_list.shape[0]): # books_list.shape[0]
# 	print('round: '+ str(i)+'/'+str(books_list.shape[0]))
# 	# year
# 	if books_original_idx[i]==-1:
# 		books_list[i, 2] = avg_year
# 	else:
# 		books_list[i, 2] = books_year[books_original_idx[i]][0]
# 	# publisher part
# 	# fill the publisher number
# 	# publisher = math.inf
# 	# if books_original_idx[i]==-1:
# 	# 	# other publisher
# 	# 	# books_list[i, 3] = str(publisher_list.shape[0])
# 	# 	# continue
# 	# 	publisher = publisher_list.shape[0]
# 	# else:
# 	# 	publisher = books_info[books_original_idx[i], 4][0]
# 	# 	if type(publisher) != str and math.isnan(publisher):
# 	# 		# other publisher
# 	# 		# books_list[i, 3] = str(publisher_list.shape[0])
# 	# 		# continue
# 	# 		publisher = publisher_list.shape[0]
# 	# 	else:
# 	# 		publisher = re.sub('[^a-zA-Z]', '', publisher).lower()
# 	# 		publisher = np.where(publisher_list==publisher)[0]
# 	# 		if publisher.shape[0] == 0:
# 	# 			# not listed
# 	# 			# other publisher
# 	# 			publisher = publisher_list.shape[0]
# 	# 		else:
# 	# 			publisher = publisher[0]
# 	# books_list[i, 3] = str(publisher)
# 	# find the dist
# 	# if books_original_idx[i]==-1:
# 	# 	pass
# 	# else:
# 	# 	publisher = books_info[books_original_idx[i], 4][0]
# 	# 	if type(publisher) != str and math.isnan(publisher):
# 	# 		publisher = ''
# 	# 	publisher = re.sub('[^a-zA-Z]', '', publisher).lower()
# 	# 	if publisher_box[publisher]==[]:
# 	# 		publisher_box[publisher] = 1
# 	# 	else:
# 	# 		publisher_box[publisher] += 1
# 	# 	count+=1

# # report the publisher dist
# # publisher_c = Counter(publisher_box)
# # with codecs.open('publisher_dist.csv', 'w', "utf-8-sig") as f:
# # 	f.write('publisher,distribution\n')
# # 	for publisher, dist in publisher_c.most_common(1000):
# # 		f.write(publisher+','+str(dist/count)+'\n')

# # save book info
# csv_columns = ['new_bookID', 'ISBN', 'year', 'original_idx']
# pd.DataFrame(books_list, columns=csv_columns).to_csv('corresponding_book_info.csv', sep=',', index=False)

# # try:
# # 	start = 40000
# # 	shift = 20000
# # 	for i in range(start, start+shift): # books_list.shape[0]
# # 		print('round: '+ str(i)+'/'+str(start+shift))
# # 		idx_in_info = math.inf
# # 		for j in range(0, books_info.shape[0]):
# # 			if books_list[i, 1] == books_info[j, 0]:
# # 				idx_in_info = j
# # 		if math.isinf(idx_in_info):
# # 			original_idx.append(-1)
# # 		else:
# # 			original_idx.append(idx_in_info)
# # except Exception as e:
# # 	pass
# # finally:
# # 	original_idx = np.array(original_idx)
# # 	pd.DataFrame(original_idx, columns=['original_idx']).to_csv('books_original_idx_'+str(start+shift)+'.csv', sep=',', index=False)

time_end = time.time()
print('execution time:', time_end - time_start, 's')
print('program ends')