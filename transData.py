import pandas as pd
import csv
import numpy as np

book_rating_train = pd.read_csv('book_ratings_train.csv')
book_rating_train = book_rating_train.values
print(book_rating_train.shape)

book_rating_test = pd.read_csv('book_ratings_test.csv')
book_rating_test = book_rating_test.values
print(book_rating_test.shape)

record_userID = []
record_user_couple = []
record_bookID = []
record_book_couple = []
user_count = 1
book_count = 1

for i in range(1, 10000):             #len(book_rating_train)+1
    userID = book_rating_train[i-1, 0]
    bookID = book_rating_train[i-1, 1]
    if userID not in record_userID:
        record_userID.append(userID)
        record_user_couple.append([userID, user_count])
        user_count = user_count + 1

    else:
        for j in range(len(record_user_couple)):
            if userID == record_user_couple[j][0]:
                record_user_couple.append([userID, record_user_couple[j][1]])
                break

    if bookID not in record_bookID:
        record_bookID.append(bookID)
        record_book_couple.append([bookID, book_count])
        book_count = book_count + 1

    else:
        for j in range(len(record_book_couple)):
            if bookID == record_book_couple[j][0]:
                record_book_couple.append([bookID, record_book_couple[j][1]])
                break

record_userID_test = []
record_user_index_test = []
record_user_couple_test = []
record_bookID_test = []
record_book_index_test = []
record_book_couple_test = []

for i in range(1, 10000):             #len(book_rating_train)+1
    userID_test = book_rating_test[i-1, 0]
    bookID_test = book_rating_test[i-1, 1]
    if userID_test not in record_userID_test and userID_test not in record_userID:
        record_userID_test.append(userID_test)
        record_user_couple_test.append([userID_test, user_count])
        user_count = user_count + 1

    else:
        user_flag = False
        for j in range(len(record_user_couple_test)):
            if userID_test == record_user_couple_test[j][0]:
                record_user_couple_test.append([userID_test, record_user_couple_test[j][1]])
                user_flag = True
                break

        if not user_flag:
            for j in range(len(record_user_couple)):
                if userID_test == record_user_couple[j][0]:
                    record_user_couple_test.append([userID_test, record_user_couple[j][1]])
                    break

    if bookID_test not in record_bookID_test and bookID_test not in record_bookID:
        record_bookID_test.append(bookID_test)
        record_book_couple_test.append([bookID_test, book_count])
        book_count = book_count + 1

    else:
        book_flag = False
        for j in range(len(record_book_couple_test)):
            if bookID_test == record_book_couple_test[j][0]:
                record_book_couple_test.append([bookID_test, record_book_couple_test[j][1]])
                book_flag = True
                break
        if not book_flag:
            for j in range(len(record_book_couple)):
                if bookID_test == record_book_couple[j][0]:
                    record_book_couple_test.append([bookID_test, record_book_couple[j][1]])
                    break

record_user_couple = np.array(record_user_couple)
new_userID = np.array(record_user_couple[:, 1])

record_book_couple = np.array(record_book_couple)
new_bookID = np.array(record_book_couple[:, 1])

ratings = np.array(book_rating_train[0:9999, 2])

new_train = np.c_[new_userID, new_bookID, ratings]

np.savetxt('new_trainData.csv', new_train, delimiter=',', fmt='%s'+',%s'+',%d')

record_user_couple_test = np.array(record_user_couple_test)
new_userID_test = np.array(record_user_couple_test[:, 1])

record_book_couple_test = np.array(record_book_couple_test)
new_bookID_test = np.array(record_book_couple_test[:, 1])

new_test = np.c_[new_userID_test, new_bookID_test]

np.savetxt('new_testData.csv', new_test, delimiter=',', fmt='%s'+',%s')