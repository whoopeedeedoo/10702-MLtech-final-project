import numpy as np, time
import sklearn
# import matplotlib.pyplot as plt
import pandas as pd
import keras
from keras.utils import np_utils
from keras.optimizers import SGD, Adam, RMSprop
from keras.losses import mean_absolute_error
from keras.layers import *
from keras.callbacks import ModelCheckpoint
# old keras?
from keras.models import Sequential, Model
import tensorflow as tf
import keras.backend as K
from keras.preprocessing import sequence





class WeightedAvgOverTime(Layer):
	def __init__(self, **kwargs):
		self.supports_masking = True
		super(WeightedAvgOverTime, self).__init__(**kwargs)
	def call(self, x, mask=None):
		if mask is not None:
			pass
			mask = K.cast(mask, K.floatx())
			mask = K.expand_dims(mask, axis=-1)
			s = K.sum(mask, axis=1)
			if K.equal(s, K.zeros_like(s)) is None:
				return K.mean(x, axis=1)
			else:
				return K.cast(K.sum(x * mask, axis=1) / K.sqrt(s), K.floatx())
		else:
			return K.mean(x, axis=1)
	def compute_output_shape(self, input_shape):
		return (input_shape[0], input_shape[-1])
	def compute_mask(self, x, mask=None):
		return None
	def get_config(self):
		base_config = super(WeightedAvgOverTime, self).get_config()
		return dict(list(base_config.items()))


















def get_bag(D):
	bag = []
	N = D.shape[0]; num_itera = 1
	for itera in range(0, num_itera):
		seed = np.random.seed()
		np.random.seed(seed)
		idx = np.random.permutation(list(range(0, N)))
		bag.append(D[idx])
	return bag
def bootstrap(D):
	N = D.shape[0]; num_record = int(N*1.05)# 7.5
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
	infos = []
	for i in range(0, usersID.shape[0]):
		# age, location, ...
		infos.append(users_info[int(usersID[i]), 2:])
	return np.array(infos)

time_start = time.time()
params = {'num_users':77805, 'num_books': 185973}

book_rating_train = pd.read_csv('data/new_trainData.csv').values
book_rating_train[:, :2] -= 1
# print(book_rating_train.shape)

book_rating_test = pd.read_csv('data/new_testData.csv').values
book_rating_test[:, :2] -= 1

users_info = pd.read_csv('data/corresponding_user_info.csv').values
users_info[:, 0] -= 1
# one-hot encoding: location
left_users_info = users_info[:, :3]
right_users_info = users_info[:, 4:]
locations = users_info[:, 3].reshape(-1, 1)
locations = np_utils.to_categorical(locations, np.max(locations)+1)
users_info = np.concatenate([left_users_info, locations, right_users_info], axis=1)
locations = None; left_users_info = None; right_users_info = None
# don't include user_csv_index
users_info = users_info[:, :-1]
# normalize info (age, ...)
left_users_info = users_info[:, :2]
right_users_info = users_info[:, 2:].astype(float)
right_users_info = (right_users_info - np.mean(right_users_info, axis=0))/(np.std(right_users_info, axis=0)+1e-8)
users_info = np.concatenate([left_users_info, right_users_info], axis=1)
left_users_info = None; right_users_info = None
# # reading records
# # read_records = pd.read_csv('data/users_read_books.csv').values
# users_read_list = [[] for user in range(0, 77805)]
# books_read_list = [[] for user in range(0, 185973)]
# # user (or book) index starts from 0
# # read_records -= 1
# # book_rating_train[:, :2]
# for record in book_rating_train[:, :2]:
# 	# user (or book) record embedding matrix index 0 是補短用的
# 	# 所以整體向後挪移1格
# 	user = record[0]; book = record[1]
# 	users_read_list[user].append(book+1)
# 	books_read_list[book].append(user+1)
# # 補短(補0)
# users_read_list = sequence.pad_sequences(users_read_list)
# books_read_list = sequence.pad_sequences(books_read_list)

# books info
# new_bookID,ISBN,year,category,original_idx
books_info = pd.read_csv('data/corresponding_book_info_with_category_7_0.07-0.93.csv').values[:, :-1]
books_info = books_info[:, [0,1,2,3]]
# one-hot encoding
left_books_info = books_info[:, :3]
# right_books_info = books_info[:, 4:]
categories = books_info[:, 3].reshape(-1, 1)
categories = np_utils.to_categorical(categories, np.max(categories)+1)
books_info = np.concatenate([left_books_info, categories], axis=1)
categories = None; left_books_info = None; right_books_info = None
# remove year
# books_info = np.delete(books_info, [2, -1], axis=1)

# normalize
left_books_info = books_info[:, :2]
right_books_info = books_info[:, 2:].astype(float)
right_books_info = (right_books_info - np.mean(right_books_info, axis=0))/(np.std(right_books_info, axis=0)+1e-8)
books_info = np.concatenate([left_books_info, right_books_info], axis=1)
left_books_info = None; right_users_info = None
# training
# mean: 7.60122366939
# std: 1.84414936834



# bagging
# bag = get_bag(book_rating_train)
# normal
# bag = [book_rating_train]
count = 0
usersID_test = book_rating_test[:, 0]
users_info_test = get_users_info(usersID_test, users_info)
ISBN_test = book_rating_test[:, 1]
books_info_test = get_users_info(ISBN_test, books_info)
# D = book_rating_train
# for D in bag:
for itera in range(0, 1):
	print('round', itera)
	# D = get_bag(book_rating_train)[0]
	D = book_rating_train
	N = D.shape[0]
	val_split = 1/N
	# val data
	usersID_val = D[:int(N*val_split), 0]
	ISBN_val = D[:int(N*val_split), 1]
	users_info_val = get_users_info(usersID_val, users_info)
	books_info_val = get_users_info(ISBN_val, books_info)
	train_rating_val = D[:int(N*val_split), 2]
	# train data
	# D = bootstrap(D[int(N*val_split):])
	# D = bootstrap(D)
	D = D[int(N*val_split):]
	usersID = D[:, 0]
	ISBN = D[:, 1]
	users_info_train = get_users_info(usersID, users_info)
	books_info_train = get_users_info(ISBN, books_info)
	train_rating = D[:, 2]
	# book_rating_train = book_rating_train[int(N*val_split):]

	laten_dim = 1; num_users = params['num_users']; num_books = params['num_books']
	user = Input(shape=[1])
	book = Input(shape=[1])
	user_info = Input(shape=[users_info_train.shape[1]])
	book_info = Input(shape=[books_info_train.shape[1]])

	# test_embed = Embedding(num_users, laten_dim, embeddings_initializer='zeros')

	user_weight = Embedding(num_users, laten_dim, embeddings_initializer='zeros', embeddings_regularizer='l2')(user)
	book_weight = Embedding(num_books, laten_dim, embeddings_initializer='zeros', embeddings_regularizer='l2')(book)
	user_weight = Flatten()(user_weight)
	book_weight = Flatten()(book_weight)
	# user_weight = Dropout(0.5)(user_weight)
	# book_weight = Dropout(0.5)(book_weight)

	# user_record1 = Embedding(num_users, users_read_list.shape[1], trainable=False, weights=[users_read_list])(user)
	# user_record1 = Flatten()(user_record1)
	# user_record1 = Embedding(num_books+1, laten_dim)(user_record1)
	# # user_record1 = Flatten()(user_record1)
	# user_record1 = Dropout(0.15)(user_record1)
	# user_record1 = WeightedAvgOverTime()(user_record1)
	# user_record1 = Add()([user_weight, user_record1])


	# book_record1 = Embedding(num_books, books_read_list.shape[1], trainable=False, weights=[books_read_list])(book)
	# book_record1 = Flatten()(book_record1)
	# book_record1 = Embedding(num_users+1, laten_dim)(book_record1)
	# # book_record1 = Flatten()(book_record1)
	# book_record1 = Dropout(0.15)(book_record1)
	# book_record1 = WeightedAvgOverTime()(book_record1)
	# book_weight = Add()([book_weight, book_record1])

	# # record_bias =  keras.layers.Concatenate(axis=-1)([user_record1, book_record1])
	# # record_bias =  Add()([user_record1, book_record1])
	# # record_bias =  keras.layers.Concatenate(axis=-1)([record_bias, user_record3, book_record3])

	# user_weight = keras.layers.Concatenate(axis=-1)([user_weight, user_info])

	user_bias1 = Embedding(num_users, 1, embeddings_initializer='zeros')(user)
	book_bias = Embedding(num_books, 1, embeddings_initializer='zeros')(book)
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

	# user_info_weight = Embedding(num_users, users_info_train.shape[1], embeddings_initializer='zeros')(user)
	# book_info_weight = Embedding(num_books, books_info_train.shape[1], embeddings_initializer='zeros')(book)
	# user_info_weight = Flatten()(user_info_weight)
	# book_info_weight = Flatten()(book_info_weight)

	# user_info_bias = Dot(axes=-1)([user_info_weight, user_info])
	# book_info_bias = Dot(axes=-1)([book_info_weight, book_info])

	# ratings = Lambda(lambda x: Add()([x, keras.backend.constant(0.8, shape=(1,1))]))(ratings)
	# ratings = Lambda(lambda x: Dot(axes=-1)([x, keras.backend.constant(0.4, shape=(1,1))]))(ratings)
	# user_info_weight = Lambda(lambda x: Dot(axes=-1)([x, keras.backend.constant(1, shape=(1,1))]))(user_info_weight)
	user_bias = Add()([user_bias1, user_bias3])
	# ratings = Add()([ratings, user_bias, book_bias])
	# user_info_bias = Add()([user_info_bias1, user_info_bias2])


	# c = keras.backend.constant(0.1)
	# ratings = keras.layers.Concatenate(axis=-1)([ratings, user_info, book_info])# , record_bias
	

	
	# c = K.repeat_elements(c,rep=1126,axis=0)
	# ratings = Add()([ratings, c])

	ratings = Dot(axes=-1)([user_weight, book_weight])
	# ratings = Lambda(lambda x: x+c)(ratings)
	info = keras.layers.Concatenate(axis=-1)([user_info, book_info])
	info = Dense(units=1024, activation='sigmoid')(info)
	info = Dropout(0.5)(info)
	info = Dense(units=1, activation='relu')(info)

	ratings = keras.layers.Concatenate(axis=-1)([ratings, info])
	ratings = Dense(units=1024, activation='sigmoid')(ratings)
	ratings = Dropout(0.5)(ratings)
	ratings = Dense(units=1, activation='relu')(ratings)
	output = Add()([ratings, user_bias, book_bias])

	model = Model([user, book, user_info, book_info], output)
	# adam = Adam(lr=0.001, decay=0.0008/14) # decay=0.0008/14
	adam = Adam()
	model.compile(optimizer=adam, loss='mae', metrics=[])

	# model.summary()
	# exit()
	filepath = 'checkpoints/w-improvement-'+time.strftime("%Y-%m-%d %H%M%S", time.localtime())+'-{epoch:02d}-{loss:.5f}-{val_loss:.5f}.hdf5'
	checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, mode='min')
	callbacks_list = [checkpoint]
	matrix = model.fit([usersID, ISBN, users_info_train, books_info_train], train_rating, batch_size=1126, epochs=15, # 16 for track 2
						# validation_split=0.1,
						validation_data=([usersID_val, ISBN_val, users_info_val, books_info_val], train_rating_val),
						# callbacks=callbacks_list,
						verbose=1,)

	# load model
	# filepath = 'checkpoints/w-best.hdf5'
	# model.load_weights(filepath)
	
	y_hat = model.predict([usersID_test, ISBN_test, users_info_test, books_info_test], verbose=1)
	# y_hat[np.where(y_hat>10)[0]] = 10
	# y_hat[np.where(y_hat<1)[0]] = 1
	# for i in range(0, y_hat.shape[0]):
	# 	y = y_hat[i]
	# 	if y==1 or y==10:
	# 		continue
	# 	floor = np.floor(y)
	# 	y_hat[i] = floor+1 if y-floor>=0.4 else floor
	# vectorized
	track = 1
	# print(y_hat)
	if track == 1:
		y_floor = np.floor(y_hat)
		round_threshold = 0.4
		isRound = np.where((y_hat-y_floor)>=round_threshold)[0]
		notRound = np.where((y_hat-y_floor)<round_threshold)[0]
		y_hat[isRound] = y_floor[isRound]+1
		y_hat[notRound] = y_floor[notRound]
		y_hat[np.where(y_hat>10)[0]] = 10
		y_hat[np.where(y_hat<1)[0]] = 1
		y_hat = y_hat.astype(int)
		# compare
		from sklearn.metrics import mean_absolute_error
		# y_me = pd.read_csv(predict_filename, header=None)
		y_others = pd.read_csv('submissions/predict_1.206883.csv', header=None)
		MAE = mean_absolute_error(y_hat, y_others)
		print('MAE:', MAE)
		print('mean:', np.mean(y_hat))
		print('std:', np.std(y_hat))
		# save
		predict_filename = 'predict'+str(count)+'_report_track1.csv'
		if MAE>2:
			# np.savetxt('badMAE_'+str(MAE)+predict_filename,y_hat,delimiter=",", fmt='%d')
			count-=2 # do again
		else:
			np.savetxt(predict_filename,y_hat,delimiter=",", fmt='%d')
	else:
		# save
		predict_filename = 'predict'+str(count)+'_report_track2.csv'
		np.savetxt(predict_filename,y_hat,delimiter=",", fmt='%f')
	
	# y_true = train_rating_val
	

	count += 2
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