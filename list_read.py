# coding=utf-8
import time, pandas as pd, numpy as np, math, re, codecs
from collections import defaultdict, Counter

read_record = pd.read_csv('data/users_read_books.csv').values
users_read_list = [[] for user in range(0, 77805)]
books_read_list = [[] for user in range(0, 185973)]
# user (or book) index starts from 0
read_record -= 1
for record in read_record:
	# user (or book) feedback embedding matrix index 0 是補短用的
	# 所以整體向後挪移1格
	user = record[0]; book = record[1]
	users_read_list[user].append(book+1)
	books_read_list[book].append(user+1)

print(users_read_list)
print(books_read_list)