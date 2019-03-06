import numpy as np, time
import sklearn
import matplotlib.pyplot as plt
import pandas as pd
import keras
from keras.utils import np_utils
from keras.optimizers import SGD, Adam, RMSprop
from keras.losses import mean_absolute_error
from keras.layers import *
from keras.callbacks import ModelCheckpoint
from keras.metrics import top_k_categorical_accuracy
# old keras?
from keras.models import Sequential, Model
import tensorflow as tf
import keras.backend as K

def get_bag(D):
	bag = []
	N = D.shape[0]; num_itera = 35
	for itera in range(0, num_itera):
		# np.random.seed()
		idx = np.random.permutation(list(range(0, N)))
		bag.append(D[idx])
	return bag
def bootstrap(D):
	N = D.shape[0]; num_record = int(N*7)# 7.5
	# np.random.seed()
	# idx = np.random.permutation(list(range(0, N)))
	idx = (np.round(np.random.rand(num_record)*(N-1))).astype(int)
	return D[idx]
# def mapK(y_true, y_pred):
# 	K = 5; precisions = []; p = 0
# 	# y_sorted = np.array(y_pred).argsort()[-K:][::-1]
# 	y_pred_top_k, y_pred_ind_k = tf.nn.top_k(y_pred, K)
# 	y_true_top_k, y_true_ind_k = tf.nn.top_k(y_true, 1)
# 	for j in range(0, K):
# 		temp = tf.equal(y_pred_ind_k[:, j], y_true_ind_k[:, 0])
# 		temp = tf.cast(temp, tf.float64)
# 		p = tf.reduce_mean(temp)
# 		precisions.append(p/(j+1))
# 	return keras.backend.sum(precisions)
def track1_mae(y_true, y_pred):
	def special_round(y):
		if y==1 or y==10:
			return y
		else:
			floor = tf.floor(y)
			return floor+1 if (y-floor>=0.5) is not None else floor
	y_hat = tf.map_fn(special_round, y_pred)
	return keras.backend.mean(tf.abs(y_hat-y_true))
def get_users_info(usersID, users_info):
	# usersID = np.concatenate([usersID.reshape(-1, 1), np.zeros((usersID.shape[0], 1))], axis=1)
	info = []
	for i in range(0, usersID.shape[0]):
		# age
		info.append(users_info[int(usersID[i]), 2:])
	return np.array(info)
def top_m_categorical_accuracy(y_true, y_pred):
	return top_k_categorical_accuracy(y_true, y_pred, k=3)
time_start = time.time()
book_rating_train = pd.read_csv('data/new_trainData.csv').values
book_rating_train[:, :2] -= 1
# print(book_rating_train.shape)

# testing data
# book_rating_test = pd.read_csv('data/new_testData.csv').values
# implicit data
book_rating_test = pd.read_csv('data/new_implicit.csv').values
book_rating_test[:, :2] -= 1

users_info = pd.read_csv('data/corresponding_user_info.csv').values
users_info[:, 0] -= 1
# one-hot encoding: location
left_users_info = users_info[:, :3]
right_users_info = users_info[:, 4:]
locations = users_info[:, 3].reshape(-1, 1)
locations = np_utils.to_categorical(locations, np.max(locations)+1)
users_info = np.concatenate([left_users_info, locations, right_users_info], axis=1)
# don't include user_csv_index
users_info = users_info[:, :-1]
# normalize info (age, ...)
left_users_info = users_info[:, :2]
right_users_info = users_info[:, 2:].astype(float)
right_users_info = (right_users_info - np.mean(right_users_info, axis=0))/(np.std(right_users_info, axis=0)+1e-8)
users_info = np.concatenate([left_users_info, right_users_info], axis=1)


# training
# mean: 7.60122366939
# std: 1.84414936834

# D = book_rating_train

params = {'num_users':77805, 'num_books': 185973}

# bagging
# bag = get_bag(book_rating_train)
# normal
bag = [book_rating_train]
count = 0
usersID_test = book_rating_test[:, 0]
users_info_test = get_users_info(usersID_test, users_info)
ISBN_test = book_rating_test[:, 1]
for D in bag:
	N = D.shape[0]
	val_split = 0.1 # 1/N
	# val data
	usersID_val = D[:int(N*val_split), 0]
	ISBN_val = D[:int(N*val_split), 1]
	users_info_val = get_users_info(usersID_val, users_info)
	train_rating_val = np_utils.to_categorical(D[:int(N*val_split), 2]-1, 10)
	# train data
	# D = bootstrap(D[int(N*val_split):])
	# D = bootstrap(D)
	D = D[int(N*val_split):]
	usersID = D[:, 0]
	ISBN = D[:, 1]
	users_info_train = get_users_info(usersID, users_info)
	train_rating = np_utils.to_categorical(D[:, 2]-1, 10)
	# book_rating_train = book_rating_train[int(N*val_split):]

	laten_dim = 1; num_users = params['num_users']; num_books = params['num_books']
	user = Input(shape=[1])
	book = Input(shape=[1])
	user_info = Input(shape=[users_info_train.shape[1]])

	user_weight = Embedding(num_users, laten_dim)(user)
	book_weight = Embedding(num_books, laten_dim)(book)
	user_weight = Flatten()(user_weight)
	book_weight = Flatten()(book_weight)

	# user_weight = keras.layers.Concatenate(axis=-1)([user_weight, user_info])

	user_bias1 = Embedding(num_users, 1)(user)
	book_bias = Embedding(num_books, 1)(book)
	# user_info_bias = Embedding(num_users, users_info_train.shape[1])(user)

	user_bias1 = Flatten()(user_bias1)
	book_bias = Flatten()(book_bias)
	# user_info_bias = Flatten()(user_info_bias)

	user_bias3 = Dot(axes=-1)([user_bias1, user_bias1])
	user_bias3 = Dot(axes=-1)([user_bias3, user_bias1])
	user_bias1 = Lambda(lambda x: Dot(axes=-1)([x, keras.backend.constant(3, shape=(1,1))]))(user_bias1)
	# user_bias2 = Dot(axes=-1)([user_bias2, user_bias2a])
	user_bias3 = Lambda(lambda x: Dot(axes=-1)([x, keras.backend.constant(25, shape=(1,1))]))(user_bias3)
	# book_bias = Lambda(lambda x: Dot(axes=-1)([x, keras.backend.constant(0.85, shape=(1,1))]))(book_bias)
	# user_info_bias = Dot(axes=-1)([user_info_bias, user_info])
	# user_info_bias1 = Lambda(lambda x: Dot(axes=-1)([x, keras.backend.constant(-2, shape=(1,users_info_train.shape[1]))]))(user_info)
	# user_info_bias2 = Lambda(lambda x: Dot(axes=-1)([x, x]))(user_info)

	ratings = Dot(axes=-1)([user_weight, book_weight])
	# ratings = Lambda(lambda x: Add()([x, keras.backend.constant(0.8, shape=(1,1))]))(ratings)
	# ratings = Lambda(lambda x: Dot(axes=-1)([x, keras.backend.constant(0.4, shape=(1,1))]))(ratings)
	# user_info_weight = Lambda(lambda x: Dot(axes=-1)([x, keras.backend.constant(1, shape=(1,1))]))(user_info_weight)
	# user_bias = Add()([user_bias1, user_bias3])
	# ratings = Add()([ratings, user_bias, book_bias])
	# user_info_bias = Add()([user_info_bias1, user_info_bias2])
	ratings = keras.layers.Concatenate(axis=-1)([ratings, user_bias1, book_bias, user_info])
	# ratings = Lambda(lambda x: Add()([x, keras.backend.constant(0.4, shape=(1,1))]))(ratings)
	# ratings = BatchNormalization()(ratings)
	ratings = Dense(units=128, activation='sigmoid')(ratings)
	ratings = Dropout(0.15)(ratings)
	# ratings = BatchNormalization()(ratings)
	ratings = Dense(units=64, activation='relu')(ratings)
	ratings = Dropout(0.15)(ratings)
	output = Dense(units=10, activation='softmax')(ratings)
	model = Model([user, book, user_info], output)
	# adam = Adam(lr=0.001, decay=0.001/2)
	adam = Adam()
	model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=[top_m_categorical_accuracy, 'acc'])

	model.summary()
	# exit()
	filepath = 'checkpoints/w-improvement-'+time.strftime("%Y-%m-%d %H%M%S", time.localtime())+'-{epoch:02d}-{acc:.5f}-{val_acc:.5f}.hdf5'
	checkpoint = ModelCheckpoint(filepath, monitor='val_top_m_categorical_accuracy', verbose=1, save_best_only=False, mode='max')
	callbacks_list = [checkpoint]
	matrix = model.fit([usersID, ISBN, users_info_train], train_rating, batch_size=1126, epochs=400,
						# validation_split=0.1,
						validation_data=([usersID_val, ISBN_val, users_info_val], train_rating_val),
						# callbacks=callbacks_list,
						verbose=1,)

	# load model
	# filepath = 'checkpoints/w-best.hdf5'
	# model.load_weights(filepath)
	
	y_hat = model.predict([usersID_test, ISBN_test, users_info_test], verbose=1)
	valid_idx = np.where(np.max(y_hat, axis=1)>=0.75)[0]
	print(valid_idx)
	print(valid_idx.shape)

	book_rating_test = book_rating_test[valid_idx]
	y_hat = np.argmax(y_hat[valid_idx], axis=1)+1
	predict_filename = 'predict'+str(count+0)+'.csv'
	# testing part
	# np.savetxt(predict_filename,y_hat.astype(int),delimiter=",", fmt='%d')
	# implicit part
	book_rating_test[:, :2] += 1
	implicit_content = np.concatenate([book_rating_test, y_hat.astype(int).reshape(-1,1)], axis=1)
	np.savetxt(predict_filename,implicit_content,delimiter=",", fmt='%d'+',%d'*2)
	# print('mean:', np.mean(y_hat))
	# print('std:', np.std(y_hat))
	exit()
	
	# y_true = train_rating_val
	
	# compare
	from sklearn.metrics import mean_absolute_error
	y_me = pd.read_csv(predict_filename, header=None)
	y_others = pd.read_csv('submissions/predict_1.206883.csv', header=None)
	MAE = mean_absolute_error(y_me, y_others)
	print('MAE:', MAE)
	count += 1
	# release memory
	model = None
	K.clear_session()


# pd.Series(matrix.history['loss']).plot(logy=True)
# pd.Series(matrix.history['val_loss']).plot(logy=True)
# plt.xlabel("Epoch")
# plt.ylabel("Train Error")
# plt.legend(['train', 'test'], loc='upper right')
# plt.show()


# print('MAE:', MAE)
# print(y_hat.shape)
# print(y_hat)
# print(y_true)
time_end = time.time()
print('execution time:', time_end - time_start, 's')
print('program ends')