import numpy as np
import pandas as pd
import math

book_rating_train = pd.read_csv('data/book_ratings_train.csv')
book_rating_train = book_rating_train.values

new_train = pd.read_csv('data/new_trainData.csv')
new_train = new_train.values
print(new_train)

book_rating_test = pd.read_csv('data/book_ratings_test.csv')
book_rating_test = book_rating_test.values

new_test = pd.read_csv('data/new_testData.csv')
new_test = new_test.values

user_info = pd.read_csv('data/users.csv')
user_info = user_info.values
print(user_info.shape)
# userID = np.zeros((len(user_info),))

max_order_user = max(new_test[:, 0])
max_order_book = max(new_test[:, 1])
# print(max_order_book)

max_order_user_train = max(new_train[:, 0])
max_order_book_train = max(new_train[:, 1])

user_list = np.array([x for x in range(1, max_order_user + 1)]).reshape(-1)
temp = np.empty((max_order_user,), dtype='str')
user_list = np.c_[user_list, temp]

# print(user_list[0,0])
book_list = np.array([x for x in range(1, max_order_book + 1)]).reshape(-1)
temp = np.empty((max_order_book,), dtype='str')
book_list = np.c_[book_list, temp]

# user_train_list = np.array([x for x in range(1, max(new_train[:, 0]) + 1)]).reshape(-1)
# temp = np.empty((max(new_train[:, 0]),), dtype='str')
# user_train_list = np.c_[user_train_list, temp]
#
# # print(user_list[0,0])
# book_train_list = np.array([x for x in range(1, max(new_train[:, 1]) + 1)]).reshape(-1)
# temp = np.empty((max(new_train[:, 1]),), dtype='str')
# book_train_list = np.c_[book_train_list, temp]

# print(book_list)
for i in range(len(new_train)):
    print('round train', i)
    user = int(new_train[i, 0])
    user_list[user - 1, 1] = book_rating_train[i, 0]

    book = int(new_train[i, 1])
    # print(book)
    book_list[book - 1, 1] = book_rating_train[i, 1]

for i in range(len(new_test)):
    print('round test', i)
    user = int(new_test[i, 0])
    user_list[user - 1, 1] = book_rating_test[i, 0]

    book = int(new_test[i, 1])
    # print(book)
    book_list[book - 1, 1] = book_rating_test[i, 1]

record = []

# for i in range(len(user_info)):               # len(user_info)
#     print('round user_info', i)
#     temp = user_list[np.where(user_list[0:max_order_user_train, 1] == user_info[i, 0]), 0]
#     print(temp, i)
#     if temp[0].size !=0 :                 # return is tuple((x,))
#         # print('not empty')
#         record.append([temp[0].item(0), user_info[i,1].replace(',',''), user_info[i,2]])

# print(record)
    # else:
    #     user_info = np.delete(user_info, i, 0)
# print(user_list)
# print(type(book_list[8,1]))
# np.savetxt('corresponding_user.csv', user_list, delimiter=',', fmt='%s'+',%s')
np.savetxt('data/corresponding_book.csv', book_list, delimiter='{', fmt='%s'+'{%s')
# np.savetxt('new_user.csv', record,  delimiter=',', fmt='%s'+',%s'+',%s')

# np.savetxt('corresponding_test_user.csv', user_list[max(new_train[:, 0]):, :], delimiter=',', fmt='%s'+',%s')
# np.savetxt('corresponding_test_book.csv', book_list[max(new_train[:, 1]):, :], delimiter=',', fmt='%s'+',%s', encoding='utf-8')
