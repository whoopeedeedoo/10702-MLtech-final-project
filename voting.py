import numpy as np, time
import pandas as pd
import sklearn
import sys, os, re


time_start = time.time()
num_voter = int(sys.argv[1]); ys = []; y_final = []
# y_template = pd.read_csv('predict.csv', header=None).values
# N = y_template.shape[0]
predict_csv_list = []
for filename in os.listdir('./'):
	if re.match(r'predict\d+\.csv', filename) != None:
		# number = re.sub('[^0-9]', '', filename)
		# if int(number)%2 == 0:
		# 	predict_csv_list.append(filename)
		predict_csv_list.append(filename)
num_voter = len(predict_csv_list) if len(predict_csv_list)<num_voter else num_voter
for i in range(0, num_voter):
	print(chr(8)*50 + 'loading round '+str(i+1)+'/'+str(num_voter), end='')
	# predict_filename = 'predict'+str(i)+'.csv'
	predict_filename = predict_csv_list[i]
	y_hat = pd.read_csv(predict_filename, header=None).values
	ys.append(y_hat)
	N = y_hat.shape[0]

# float input version
# y_final = []
# for i in range(0, num_voter):
# 	if y_final == []:
# 		y_final = ys[i]
# 	else:
# 		y_final += ys[i]
# y_final /= num_voter
# # vectorized
# y_floor = np.floor(y_final)
# isRound = np.where((y_final-y_floor)>=0.42)[0]
# notRound = np.where((y_final-y_floor)<0.42)[0]
# y_final[isRound] = y_floor[isRound]+1
# y_final[notRound] = y_floor[notRound]
# y_final[np.where(y_final>10)[0]] = 10
# y_final[np.where(y_final<1)[0]] = 1

# int input version
print('')
for i in range(0, N):
	# clear line
	console_col = 5
	print(chr(8)*50 + 'voting round '+str(i+1)+'/'+str(N), end='')
	# sys.stdout.flush()
	# init
	vote_box = {}
	for score in range(1, 10+1):
		vote_box[score] = 0
	# voting
	# preloading version
	for idx in range(0, num_voter):
		vote = ys[idx][i, 0]
		vote_box[vote] += 1
	# memory-saving version
	# for idx in range(0, num_voter):
	# 	predict_filename = 'predict'+str(idx)+'.csv'
	# 	y_hat = pd.read_csv(predict_filename, header=None).values
	# 	vote = y_hat[i, 0]
	# 	vote_box[vote] += 1
	# find winner
	winner = -1; votes = -1
	for score in range(1, 10+1):
		if vote_box[score] >= votes:
			winner = score
			votes = vote_box[score]
	y_final.append(winner)

print('')
# save and compare
np.savetxt('predict.csv', y_final, delimiter=",", fmt='%d')
# compare
from sklearn.metrics import mean_absolute_error
y_me = pd.read_csv('predict.csv', header=None)
y_others = pd.read_csv('submissions/predict_1.206883.csv', header=None)
MAE = mean_absolute_error(y_me, y_others)
print('MAE:', MAE)
time_end = time.time()
print('execution time:', time_end - time_start, 's')
print('program ends')

# k = round((i/len(data))*100)
#     sys.stdout.write("\r[%-50s] %d%%" %('>'*int(k/2) ,k))
#     sys.stdout.flush()
