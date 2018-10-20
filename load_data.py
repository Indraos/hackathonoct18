import pandas as pd
import pickle
import numpy as np
import datetime
import os
from os import listdir
from os.path import isfile, join
import dateutil.parser as pa


path_labels = '../StocksMinute'
path_data = '../Reuters_US'

def load_labels(path, subcompany_list=['AAPL']):
	data_dict = {fn.split('_')[0]: pd.DataFrame(columns=['date','time','close']) for fn in os.listdir(path)}
	for filename in os.listdir(path):
		ticker = filename.split('_')[0]
		if ticker in subcompany_list:
			try:
				data_dict[ticker] = data_dict[ticker].append(load_file(filename, path))
			except pd.io.common.EmptyDataError:
				print(filename, " is empty and has been skipped.")
	for i in data_dict:
		if 'time' in data_dict[i].columns:
			data_dict[i].drop('time', axis=1, inplace=True)
	return data_dict

def load_file(filename, path):
	data = pd.read_csv(path + '/' + filename, usecols=[2,3,7], dtype=str)
	data.columns = ['date','time','close']
	data['date'] = (data['date'].map(str) + data['time'].map(str)).map(transform_time)
	data = data.dropna(axis=0)
	return data

def transform_time(timestring):
	return datetime.datetime(int(timestring[:4]), int(timestring[4:6]), int(timestring[6:8]), int(timestring[8:10]), int(timestring[10:12]))

def transform_data(data, diff):
	for key in data:
		data[key].set_index(pd.DatetimeIndex(data[key]['date']),inplace=True)
		data[key]['close'] = pd.to_numeric(data[key]['close'])
		data[key]['target'] = data[key]['close'].diff(diff) > 0
		data[key].drop(['close','date'], axis=1, inplace=True)
		return data

def transform_time_text(timestring):
	if timestring != 'nan' and timestring != "" and timestring != 'NaT' and 'endif' not in timestring:
		a = timestring.split(' ')
		try:
			hours = a[3].split(':')
		except ValueError as error:
			print(a,timestring)
		month_dict = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
		return datetime.datetime(int(a[5]), int(month_dict[a[1]]), int(a[2]), int(hours[0]), int(hours[1]))


def load_sentences(path, subset=['reuters_us_news_20171010.csv']):
	data = pd.DataFrame(columns=['date','abstract','text'])
	for filename in os.listdir(path):
		if filename in subset:
			print(filename)
			data = data.append(pd.read_csv(path + '/' + filename, sep=';', usecols=[0,1,2], names=['date','abstract','text']))
	data['date'] = data['date'].map(str).map(transform_time_text)
	data.set_index(pd.DatetimeIndex(data['date']), inplace=True)
	return data

def join_data(path_labels, path_data, subset):
	labels = transform_data(load_labels(path_labels), 30)['AAPL'].sort_index()
	data = load_sentences(path_data, subset).sort_index()
	return labels, data, pd.merge(data,labels, how='inner', left_index=True, right_index=True)



YEAR = '2017'
for i in range(1, 13, 1):
	this_year_and_month = YEAR + str(i).zfill(2)
	all_paths = [f for f in listdir(path_data) if isfile(join(path_data, f))]
	paths_from_one_month = [path for path in all_paths if this_year_and_month in path]
	print(paths_from_one_month)
	_,_,merde_data = join_data(path_labels,path_data, paths_from_one_month)
	print(merde_data)