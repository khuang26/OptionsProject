import pandas as pd
import numpy as np
import math
from scipy.stats import norm


class OptionsData: 
	def __init__(self, fileName):
		self.fileName = fileName
		self.df = pd.read_csv(fileName)
		self.__calls = self.df[self.df['is_call'] == 1]
		self.__puts = self.df[self.df['is_call'] == 0]
		self.__pairs = None
		self.__avgRate = None
		self.__rateStDev = None
		self.__interest_rates = None


	def __findPairs(self, minTime = 0, maxTime = 28800, maxTimeGap = 28800): 
		self.__pairs = pd.merge(
			self.__calls[['seconds','symbol', 'strike_price', 'year_to_expiration', 'u_trade_px', 'trade_price', 'trade_qty']],
			self.__puts[['seconds', 'symbol', 'strike_price', 'year_to_expiration', 'u_trade_px', 'trade_price', 'trade_qty']],
			on=['strike_price', 'year_to_expiration', 'u_trade_px'], # Columns that need to match
			suffixes=('_call', '_put')
		)
		
		self.__pairs = self.__pairs.groupby([
			'seconds_call', 
			'seconds_put',
			'strike_price', 
			'year_to_expiration', 
			'u_trade_px', 
			'trade_price_call', 
			'trade_price_put', 
			'symbol_call', 
			'symbol_put'
		], as_index=False).agg({
			'trade_qty_call': 'sum',
			'trade_qty_put': 'sum'
		})

		mask = self.__pairs.apply(
			lambda row: self.__checkRowValidity(row, minTime, maxTime, maxTimeGap), 
			axis=1) 
		self.__pairs = self.__pairs[mask] 


		self.__pairs['trade_qty'] = self.__pairs['trade_qty_call'] + self.__pairs['trade_qty_put']
		
		self.__pairs['seconds'] = (self.__pairs['seconds_call'] + self.__pairs['seconds_put'])/2 # Should change this to be max()

	def computeRFRate(self, minTime = 0, maxTime = 28800, maxTimeGap = 600):
		self.__findPairs(minTime, maxTime, maxTimeGap)
		self.__computeSeriesRates()
		
		weighted_sum = np.dot(self.__pairs['rate'], self.__pairs['trade_qty'])
		total_qty = self.__pairs['trade_qty'].sum()
		weighted_avg_rate = weighted_sum / total_qty
		

		weighted_var = ((self.__pairs['rate'] - weighted_avg_rate)**2 * self.__pairs['trade_qty']).sum() / total_qty
		weighted_StDev = np.sqrt(weighted_var)

		if (minTime == 0) and (maxTime == 28800) and (maxTimeGap == 600): 
			self.setAvgRate(weighted_avg_rate)
			self.__rateStDev = weighted_StDev

		print(f'The average risk-free rate across all valid put-call pairs occurring between {minTime} and {maxTime} seconds with a max time difference of {maxTimeGap} seconds is {weighted_avg_rate}, and the standard deviation is {weighted_StDev}.')
		self.__writePairs()

		return weighted_avg_rate
	
	def __writePairs(self):
		self.__pairs.to_csv('Pairs.csv', index=False)

	def __checkRowValidity(self, row, minTime = 0, maxTime = 28800, maxTimeGap = 600):
		return not (
			(abs(row['seconds_call'] - row['seconds_put']) > maxTimeGap) 
			or (row['seconds_call'] > maxTime) 
			or (row['seconds_put'] > maxTime)
			or (row['seconds_call'] < minTime)
			or (row['seconds_put'] < minTime)
    	)


	def __computeRate(self, row): 
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
		self.__pairs['rate'] = self.__pairs.apply(
			self.__computeRate, 
			axis = 1) 
		
	def computeGreeks(self): 
		self.df['delta'] = self.df.apply(
			self.__computeDelta, 
			axis = 1)
		self.df['gamma'] = self.df.apply(
			self.__computeGamma, 
			axis = 1)
		self.df['vega'] = self.df.apply(
			self.__computeVega, 
			axis = 1)
		self.df['theta'] = self.df.apply(
			self.__computeTheta, 
			axis = 1)
		self.df['rho'] = self.df.apply(
			self.__computeRho, 
			axis = 1)
		print("Finished computing Greeks!")

	def getAvgRate(self):
		return self.__avgRate
	
	def setAvgRate(self, rate): 
		oldRate = self.__avgRate
		self.__avgRate = rate
		return oldRate

	@classmethod
	def __computeBSP(cls, S, K, T, r, sigma, is_call):
		d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
		d2 = d1 - sigma * np.sqrt(T)

		if is_call: 
			price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
		else: 
			price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

		return price
	
	"""def __computeBSPs(self):
		self.df.['BSP']
		for idx, row in self.df.iterrows(): """
			
	def __createInterestRateTable(self): 
		self.__interest_rates = []
		for i in range(0, 32): 
			self.__interest_rates.append(self.computeRFRate(900*i, 900*(i+1), 600))

	def computeImpliedVolatilities(self, maxIterations = 100, tolerance = 0.005): 
		self.df['implied_vol'] = np.nan

		self.__createInterestRateTable()

		for idx, row in self.df.iterrows():
			S = row['u_trade_px']
			K = row['strike_price']
			T = row['year_to_expiration']
			t = row['seconds']
			if t>=900: 
				r = self.__interest_rates[math.floor(t/900) - 1]
			else: 
				continue
			market_price = row['trade_price']

			is_call = (row['is_call'] == 1)

			iv = 0.3

			error = OptionsData.__computeBSP(S, K, T, r, iv, is_call) - market_price

			# f(x) = BSP - market_price (latter term is constant, so it disappears when taking derivative)
			def __vega(): 
				d1 = (np.log(S / K) + (r + iv**2 / 2) * T) / (iv * np.sqrt(T))
				return S * np.sqrt(T) * norm.pdf(d1)

			i = 0
			while (abs(error) > tolerance) and (i <= maxIterations): 
				iv -= error/__vega()
				error = OptionsData.__computeBSP(S, K, T, r, iv, is_call) - market_price
			

			self.df.at[idx, 'implied_vol'] = iv
		
		print("IV calculations done!")

    # Greeks
	def __computeDelta(self ,row): 
		S = row['u_trade_px']
		K = row['strike_price']
		T = row['year_to_expiration']
		sigma = row['implied_vol']
		t = row['seconds']
		is_call = (row['is_call'] == 1)
		
		if t>=900: 
			r = self.__interest_rates[math.floor(t/900) - 1]
		else: 
			return np.nan

		d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
		d2 = d1 - sigma * np.sqrt(T)

		if is_call: 
			delta = norm.cdf(d1)
		else: 
			delta = norm.cdf(d1) - 1
		
		return delta

	def __computeGamma(self, row): 
		S = row['u_trade_px']
		K = row['strike_price']
		T = row['year_to_expiration']
		sigma = row['implied_vol']
		t = row['seconds']
		is_call = (row['is_call'] == 1)

		if t>=900: 
			r = self.__interest_rates[math.floor(t/900) - 1]
		else: 
			return np.nan
		
		d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
		d2 = d1 - sigma * np.sqrt(T)
		
		gamma = norm.pdf(d1) * 1/(S * sigma * np.sqrt(T))
		return gamma
	
	def __computeVega(self, row): 
		S = row['u_trade_px']
		K = row['strike_price']
		T = row['year_to_expiration']
		sigma = row['implied_vol']
		t = row['seconds']
		is_call = (row['is_call'] == 1)

		if t>=900: 
			r = self.__interest_rates[math.floor(t/900) - 1]
		else: 
			return np.nan
		
		d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
		d2 = d1 - sigma * np.sqrt(T)
		
		vega = S * norm.pdf(d1) * np.sqrt(T)
		return vega
	
	def __computeTheta(self, row): 
		S = row['u_trade_px']
		K = row['strike_price']
		T = row['year_to_expiration']
		sigma = row['implied_vol']
		t = row['seconds']
		is_call = (row['is_call'] == 1)

		if t>=900: 
			r = self.__interest_rates[math.floor(t/900) - 1]
		else: 
			return np.nan
		
		d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
		d2 = d1 - sigma * np.sqrt(T)
		
		if is_call: 
			theta = S * norm.pdf(d1) * sigma / (2*np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(d2)
		else: 
			theta = S * norm.pdf(d1) * sigma / (2*np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(-d2)
		return theta
		
	def __computeRho(self, row): 
		S = row['u_trade_px']
		K = row['strike_price']
		T = row['year_to_expiration']
		sigma = row['implied_vol']
		t = row['seconds']
		is_call = (row['is_call'] == 1)

		if t>=900: 
			r = self.__interest_rates[math.floor(t/900) - 1]
		else: 
			return np.nan
		
		d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
		d2 = d1 - sigma * np.sqrt(T)
		
		if is_call: 
			rho = T * K * np.exp(-r*T) * norm.cdf(d2)
		else: 
			rho = -T * K * np.exp(-r*T) * norm.cdf(-d2)
		return rho


	def writeDF(self): 
		self.df.to_csv('OptionsData.csv', index=False)




vale = OptionsData('Options Data - data.csv')
"""for i in range(1, 9): 
	vale.computeRFRate(3600*(i-1), 3600*i, 600)
vale.computeRFRate(0, 28800, 28800)
vale.computeRFRate(0, 28800, 600)"""

vale.computeImpliedVolatilities()
vale.computeGreeks()
vale.writeDF()
