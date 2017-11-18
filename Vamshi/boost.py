# -*- coding: utf-8 -*-
# @Author: vamshi
# @Date:   2017-11-18 20:03:05
# @Last Modified by:   vamshi
# @Last Modified time: 2017-11-18 20:27:29

import pandas as pd
import numpy as np
import sklearn
import datetime
import matplotlib.pyplot as plt

from matplotlib import pyplot as plt

#from tensorflow.contrib import learn
from sklearn.metrics import mean_squared_error, mean_absolute_error
#from lstm_predictor import lstm_model
import xgboost as xgb
#import lightboost as lgb

train_file = "./train_aWnotuB.csv"
test_file  = "./test_BdBKkAj.csv"

header = ["DateTime", "Junction","Vehichles", "ID"]
header_t = ["DateTime", "Junction", "ID"]

## This function takes in input the date in the format(yyyy-mm-dd) and returns the day
def tell_day(dt):
    year, month, day = (int(x) for x in dt.split('-'))
    ans = datetime.date(year, month, day)
    return ans.strftime("%w")

def read(file_dir,mode="train",header=header):
	df = pd.read_csv(file_dir,header=0,names=header)
	#plt.plot(df['Vehichles'])
	Days,Year,Month,Date,Hours = [],[],[],[],[]

	for dt in df['DateTime']:
		dt_ = dt
		dt = dt.split(" ")[0]
		time = dt_.split(" ")[1]
		hour = time.split(":")[0]
		Hours.append(int(hour))
		year, month, day = (int(x) for x in dt.split('-'))
		Year.append(int(year))
		Month.append(int(month))
		Date.append(int(day))
		day = tell_day(dt)
		Days.append(day)
	
	df['DateTime'] = df['DateTime'].astype('datetime64[ns]')
	#Add new columns
	df = pd.DataFrame(df)
	
	df = df.assign(Year=Year)
	df = df.assign(Month=Month)
	df = df.assign(Date=Date)
	df = df.assign(Hour = Hours)
	#df = df.assign(day= Days)

	del df['ID']
	del df['DateTime']

	if(mode=="train"):
		vech = df['Vehichles']
		del df['Vehichles']
		df = df.assign(Vehichles=vech)

	#rolling_mean = pd.rolling_mean(df['Vehichles'], window=7)
	#plt.plot(rolling_mean)
	#plt.show()
 	return df

def split_train(X,y):
	l = X.shape
	train_X = X[0:10000]
	train_y = y[0:10000]
	val_X = X[10000:]
	val_y = y[10000:]
	return train_X,val_X,train_y,val_y


X_test = read(test_file, mode="test",header=header_t)

df = read(train_file)
df = df.as_matrix()

X_tr,X_val, y_tr, y_val = split_train(df[:,0:4], df[:,5])


dtrain = xgb.DMatrix(X_tr,label=y_tr)
dval=xgb.DMatrix(X_val)
#dtest = xgb.DMatrix(X_test)

parameters={'max_depth':5, 'eta':0.4, 'silent':0,'objective':'regression:logistic','eval_metric':'auc','learning_rate':.05}
num_round = 500

xg = xgb.train(parameters,dtrain,num_round) 
ypred_val=xg.predict(dval)

pred = xg.predict(dtest)
