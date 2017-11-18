# -*- coding: utf-8 -*-
# @Author: vamshiteja
# @Date:   2017-11-18 07:50:22
# @Last Modified by:   vamshi
# @Last Modified time: 2017-11-18 12:34:37

import pandas as pd
import numpy as np
import sklearn
import datetime
import matplotlib.pyplot as plt


train_file = "./train_aWnotuB.csv"
test_file  = "./test_BdBkkAj.csv"

header = ["DateTime", "Junction","Vehichles", "ID"]

## This function takes in input the date in the format(yyyy-mm-dd) and returns the day
def tell_day(dt):
    year, month, day = (int(x) for x in dt.split('-'))
    ans = datetime.date(year, month, day)
    return ans.strftime("%A")

def read(file_dir):
	df = pd.read_csv(file_dir,header=0,names=header)
	#plt.plot(df['Vehichles'])
	Days = []
	for dt in df['DateTime']:
		dt = dt.split(" ")[0]
		day = tell_day(dt)
		Days.append(day)
	
	df['DateTime'] = df['DateTime'].astype('datetime64[ns]')
	df = pd.DataFrame(df)
	df = df.assign(day= Days)
	df['day'] = df['day'].astype('category')
	df['day'] = df['day'].cat.codes

	rolling_mean = pd.rolling_mean(df['Vehichles'], window=7)
	plt.plot(rolling_mean)
	plt.show()
	print rolling_mean
 	return df

df = read(train_file)
print df['DateTime'].values
ts = df