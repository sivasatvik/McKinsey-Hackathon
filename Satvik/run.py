import datetime

## This function takes in input the date in the format(yyyy-mm-dd) and returns the day
def tell_day(dt):
    year, month, day = (int(x) for x in dt.split('-'))
    ans = datetime.date(year, month, day)
    # print ans
    return ans.strftime("%w")



tell_day("2015-11-01")  # Sunday


import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6


test = 'test_BdBKkAj.csv'
train = './train_aWnotuB.csv'

def train_data(train):
    data = []
    line = 0
    file = open(train,'r')
    for row in file:
        temp = []
        line += 1
        if(line == 1):
            continue
        row = row.rstrip().split(",")
        x = row[0].split(" ")
        week_day = tell_day(x[0])
        y = x[0].split("-")
        z = x[1].split(":")
        hour = z[0]
        temp.append(int(y[0]))
        temp.append(int(y[1]))
        temp.append(int(y[2]))
        temp.append(int(week_day))
        temp.append(int(hour))
        temp.append(int(row[1]))
        temp.append(int(row[2]))
        data.append(temp)

    return data


def test_data(test):
    data = []
    line = 0
    file = open(test, 'r')
    for row in file:
        temp = []
        line += 1
        if (line == 1):
            continue
        row = row.rstrip().split(",")
        x = row[0].split(" ")
        # print x[0]
        week_day = tell_day(x[0])
        y = x[0].split("-")
        z = x[1].split(":")
        hour = z[0]
        temp.append(int(y[0]))
        temp.append(int(y[1]))
        temp.append(int(y[2]))
        temp.append(int(week_day))
        temp.append(int(hour))
        temp.append(int(row[1]))
        # temp.append(int(row[2]))
        data.append(temp)

    return data


data = train_data(train)
# print data[48119]
jn1_train_data = []
jn2_train_data = []
jn3_train_data = []
jn4_train_data = []
count = 0
for i in data:
    if(i[5] == 1):
        # count += 1
        # print (i[5])
        jn1_train_data.append(i)
    if(i[5] == 2):
        jn2_train_data.append(i)
    if(i[5] == 3):
        jn3_train_data.append(i)
    if(i[5] == 4):
        jn4_train_data.append(i)

print np.array(jn1_train_data).shape
# print jn1_train_data[48119]
# print count

j1_X_train, j1_y_train = np.array(jn1_train_data)[:,:6], np.array(jn1_train_data)[:,6]
j2_X_train, j2_y_train = np.array(jn2_train_data)[:,:6], np.array(jn2_train_data)[:,6]
j3_X_train, j3_y_train = np.array(jn3_train_data)[:,:6], np.array(jn3_train_data)[:,6]
j4_X_train, j4_y_train = np.array(jn4_train_data)[:,:6], np.array(jn4_train_data)[:,6]
print j1_X_train[0]

data_test = test_data(test)

print np.array(data_test).shape
jn1_test_data = []
jn2_test_data = []
jn3_test_data = []
jn4_test_data = []
for i in data_test:
    # print type(i[5])
    if(i[5] == 1):
        jn1_test_data.append(i)
    elif(i[5] == 2):
        jn2_test_data.append(i)
    elif(i[5] == 3):
        jn3_test_data.append(i)
    elif(i[5] == 4):
        jn4_test_data.append(i)


print jn1_test_data[0]

# from sklearn.cross_validation import train_test_split
j1_X_valid, j1_y_valid = j1_X_train[11673:], j1_y_train[11673:]
j1_X_train, j1_y_train = j1_X_train[:11673], j1_y_train[:11673]

# from sklearn.neural_network import MLPClassifier
# clf = MLPClassifier(hidden_layer_sizes=(100, ), activation='relu', solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.000000001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
# clf.fit(j1_X_train,j1_y_train)
#
# j1_y_pred = clf.predict(j1_X_valid)


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
model = Sequential()
model.add(LSTM(11674, input_shape=(1, 6)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(j1_X_train, j1_y_train)
j1_y_pred = model.predict(j1_X_valid)



print j1_y_pred
print j1_y_valid


from sklearn.metrics import mean_squared_error

print mean_squared_error(j1_y_valid, j1_y_pred)