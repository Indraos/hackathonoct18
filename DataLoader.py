import pandas as pd
import pickle
import numpy as np
import datetime
import os
from os import listdir
from os.path import isfile, join
import dateutil.parser as pa

class DataLoader:
	"""
	Class loading and joining data from Reuters and Stocks Data Set
	"""
	def __init__(self, subcompany_list=['AAPL'], diff=30, tol=.1, subset=['reuters_us_news_20171010.csv']):
		self.subcompany_list = subcompany_list
		self.diff = diff
		self.tol = tol
		self.subset = subset

	def __transform_time_reuters(self, timestring):
		return datetime.datetime(int(timestring[:4]), int(timestring[4:6]), int(timestring[6:8]), int(timestring[8:10]), int(timestring[10:12]))

	def __transform_close_prices(self, data, diff):
		data.set_index(pd.DatetimeIndex(data['date']),inplace=True)
		data['close'] = pd.to_numeric(data['close'])
		data['target'] = np.sign(data['close'].pct_change(diff))
		data.drop(['close','date'], axis=1, inplace=True)
		return data

	def __load_labels(self):
		data_dict = {fn.split('_')[0]: pd.DataFrame(columns=['date','time','close']) for fn in os.listdir(self.path_labels)}
		for filename in os.listdir(self.path_labels):
			ticker = filename.split('_')[0]
			if ticker in self.subcompany_list:
				try:
					data_dict[ticker] = data_dict[ticker].append(self.__load_file(filename))
				except pd.io.common.EmptyDataError:
					print(filename, " is empty and has been skipped.")
		for i in data_dict:
			if 'time' in data_dict[i].columns:
				data_dict[i].drop('time', axis=1, inplace=True)
			self.__transform_close_prices(data_dict[i], self.diff)
		self.labels_dict = data_dict
		return data_dict

	def __load_label(self, filename):
		data = pd.read_csv(self.path_labels + '/' + filename, usecols=[2,3,7], dtype=str)
		data.columns = ['date','time','close']
		data['date'] = (data['date'].map(str) + data['time'].map(str)).map(transform_time)
		data = data.dropna(axis=0)
		return data

	def __load_data(self):
		data = pd.DataFrame(columns=['date','abstract','text'])
		for filename in os.listdir(self.path_data):
			if filename in self.subset:
				data = data.append(pd.read_csv(self.path_data + '/' + filename, sep=';', usecols=[0,1,2], names=['date','abstract','text']))
		data['date'] = data['date'].map(str).map(self.__transform_time_reuters)
		data.set_index(pd.DatetimeIndex(data['date']), inplace=True)
		return data

	def load(self, path_data, path_labels):
		self.path_data = path_data
		self.path_labels = path_labels
		data = self.__load_data().sort_index()
		self.data = {k: pd.merge(data,v, how='inner', left_index=True, right_index=True) for k, v in self.__load_labels().items()}
		return {k: pd.merge(data,v, how='inner', left_index=True, right_index=True) for k, v in self.__load_labels().items()}



data = DataLoader()
a = data.load('/Volumes/Elements/Nachrichten und Kurse/Reuters_US', '/Volumes/Elements/Nachrichten und Kurse/StocksMinute')
