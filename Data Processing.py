import pandas as pd
import numpy as np



class OptionsData: 
	def __init__(self, fileName):
		self.fileName = fileName
		self.df = pd.read_csv(fileName)
		self.__calls = self.df[self.df['is_call'] == 1]
		self.__puts = self.df[self.df['is_call'] == 0]
		self.__pairs = None
		self.__avgRate = None
		self.__rateStDev = None

	def computeRFRate(self):
		self.__findPairs()
		self.__computeSeriesRates()
		self.setAvgRate(self.__pairs['rate'].mean())
		self.__rateStDev = self.__pairs['rate'].std()
		print(f"The average risk-free rate across all valid put-call pairs is: {self.__avgRate} and the standard deviation is {self.__rateStDev}")
		self.__writePairs()


	def __findPairs(self): 
		self.__pairs = pd.merge(
			self.__calls[['strike_price', 'year_to_expiration', 'u_trade_px', 'trade_price', 'trade_qty']],
			self.__puts[['strike_price', 'year_to_expiration', 'u_trade_px', 'trade_price', 'trade_qty']],
			on=['strike_price', 'year_to_expiration', 'u_trade_px'], # Columns that need to match
			suffixes=('_call', '_put')
		)
	
	def __printPairs(self): 
		print(self.__pairs.head())
	
	def __writePairs(self):
		self.__pairs.to_csv("Pairs.csv", index=False)

	def __getRate(self, row): 
		S = row['u_trade_px']
		K = row['strike_price']
		C = row['trade_price_call']
		P = row['trade_price_put']
		T = row['year_to_expiration']

		if T <= 0 or K <= 0 or (S - C + P) <= 0:
			return np.nan

		r = - (np.log((S - C + P) / K)) / T
		return r
		
	def __computeSeriesRates(self): 
		self.__pairs['rate'] = self.__pairs.apply(self.__getRate, axis = 1) # new column "rate" created in DataFrame; axis = 1 --> checks each row

	def getAvgRate(self):
		return self.__avgRate
	
	def setAvgRate(self, rate): 
		oldRate = self.__avgRate
		self.__avgRate = rate
		return oldRate

vale = OptionsData("Options Data - data.csv")
vale.computeRFRate()
