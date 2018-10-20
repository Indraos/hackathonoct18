import pandas as pd
import pickle
import numpy as np
import datetime
import os
from os import listdir
from os.path import isfile, join

YEAR_START = 2010
YEAR_END = 2017
class DataLoader:
	"""
	Class loading and joining data from Reuters and Stocks Data Set
	"""
	def __init__(self, subcompany_list=['AAPL'], diff=30, tol=.5, subset=['reuters_us_news_20171010.csv']):
		self.subcompany_list = subcompany_list
		self.diff = diff
		self.tol = tol
		self.subset = subset

	def __transform_time_stocks(self, timestring):
		try:
			return datetime.datetime(int(timestring[:4]), int(timestring[4:6]), int(timestring[6:8]), int(timestring[8:10]), int(timestring[10:12]))
		except ValueError as err:
			print(err)
			return pd.NaT

	def __transform_time_reuters(self, timestring):
		a = timestring.split(' ')
		try:
			hours = a[3].split(':')
			month_dict = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
			return datetime.datetime(int(a[5]), int(month_dict[a[1]]), int(a[2]), int(hours[0]), int(hours[1]))
		except IndexError as err:
			print(a)
			return pd.NaT
		except ValueError as err:
			print("Malformed date ", err)
			
	def __transform_close_prices(self, data, diff):
		data.set_index(pd.DatetimeIndex(data['date']),inplace=True)
		data['close'] = pd.to_numeric(data['close'])
		cutoff = lambda x: 1 if np.sign(x) > self.tol else 0 if np.sign(x) < -self.tol else .5
		data['target'] = data['close'].pct_change(diff).map(cutoff)
		data.drop(['close','date'], axis=1, inplace=True)
		return data

	def __load_labels(self):
		data_dict = {}
		for filename in os.listdir(self.path_labels):
			ticker = filename.split('_')[0]
			if ticker in self.subcompany_list:
				if ticker in data_dict:
					try:
						data_dict[ticker] = data_dict[ticker].append(self.__load_label(filename))
					except pd.io.common.EmptyDataError:
						print(filename, " is empty and has been skipped.")
				else:
					data_dict[ticker] = self.__load_label(filename)
		for ticker in data_dict:
			if 'time' in data_dict[ticker].columns:
				data_dict[ticker].drop('time', axis=1, inplace=True)
			self.__transform_close_prices(data_dict[ticker], self.diff)
			data_dict[ticker].columns = [ticker]
		self.labels = data_dict.values()
		return self.labels

	def __load_label(self, filename):
		data = pd.read_csv(os.path.join(self.path_labels, filename), usecols=[2,3,7], dtype=str)
		data.columns = ['date','time','close']
		data['date'] = (data['date'].map(str) + data['time'].map(str)).map(self.__transform_time_stocks)
		data = data.dropna(axis=0)
		return data

	def __load_data(self):
		data = pd.DataFrame(columns=['date','abstract','text'])
		for filename in os.listdir(self.path_data):
			if filename in self.subset:
				data = data.append(pd.read_csv(os.path.join(self.path_data,filename), sep=';', usecols=[0,1,2], names=['date','abstract','text']))
		data['date'] = data['date'].map(str).map(self.__transform_time_reuters)
		data.set_index(pd.DatetimeIndex(data['date']), inplace=True)
		return data

	def load(self, path_data, path_labels):
		self.path_data = path_data
		self.path_labels = path_labels
		labels = self.__load_labels()
		labels = pd.concat(labels)
		data = self.__load_data().sort_index()
		self.data = pd.merge(data, labels, how='inner', left_index=True, right_index=True)
		return self.data

	def save(self, path):
		for year in range(YEAR_START, YEAR_END):
			for month in range(1,13):
				filename = os.path.join(str(year),str(month))
				start_date = str(year) + '-' + str(month) + '-01'
				end_date = str(year) + '-' + str(month) + '-28'
				os.makedirs(os.path.dirname(os.path.join(path,filename)), exist_ok=True)
				with open(os.path.join(path,filename), "wb") as out_file:
				    pickle.dump(self.data.loc[start_date:end_date], out_file)






reuter_path = '../Reuters_US'
all_paths = [f for f in listdir(reuter_path) if isfile(join(reuter_path, f))]
keep_paths = []
for year in range(YEAR_START, YEAR_END):
	for i in range(1, 13, 1):
		this_year_and_month = str(year) + str(i).zfill(2)
		for path in all_paths:
			if this_year_and_month in path:
				keep_paths.append(path)


data = DataLoader(subcompany_list=['AAPL'], subset=keep_paths)
a = data.load(reuter_path, '../StocksMinute')
# data.calculate_embedding()
data.save(r'./data')