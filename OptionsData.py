import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
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

		self.__marketCloseTime = 27000

	# Interest rate calculations

	def __findPairs(self, minTime = 0, maxTime = None, maxTimeGap = None): 
		if (maxTime == None) or (maxTime > self.__marketCloseTime): 
			maxTime = self.__marketCloseTime
		if (maxTime == None) or (maxTime > self.__marketCloseTime): 
			maxTime = self.__marketCloseTime

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
		
		self.__pairs['seconds'] = self.__pairs.apply(
			self.__getMaxTime, 
			axis = 1)

		mask = self.__pairs.apply(
			lambda row: self.__checkRowValidity(row, minTime, maxTime, maxTimeGap), 
			axis=1) 
		self.__pairs = self.__pairs[mask] 


		self.__pairs['trade_qty'] = self.__pairs['trade_qty_call'] + self.__pairs['trade_qty_put']

	def computeRFRate(self, minTime = 0, maxTime = None, maxTimeGap = 600):
		if (maxTime == None) or (maxTime > self.__marketCloseTime): 
			maxTime = self.__marketCloseTime

		self.__findPairs(minTime, maxTime, maxTimeGap)
		self.__computeSeriesRates()
		
		weighted_sum = np.dot(self.__pairs['rate'], self.__pairs['trade_qty'])
		total_qty = self.__pairs['trade_qty'].sum()
		weighted_avg_rate = weighted_sum / total_qty
		

		weighted_var = ((self.__pairs['rate'] - weighted_avg_rate)**2 * self.__pairs['trade_qty']).sum() / total_qty
		weighted_StDev = np.sqrt(weighted_var)

		if (minTime == 0) and (maxTime == self.__marketCloseTime) and (maxTimeGap == 600): 
			self.setAvgRate(weighted_avg_rate)
			self.__rateStDev = weighted_StDev

		# print(f'The average risk-free rate across all valid put-call pairs occurring between {minTime} and {maxTime} seconds with a max time difference of {maxTimeGap} seconds is {weighted_avg_rate}, and the standard deviation is {weighted_StDev}.')
		self.__writePairs()

		return weighted_avg_rate

	# Interest Rate Helper Functions

	def __computeSeriesRates(self): 
		self.__pairs['rate'] = self.__pairs.apply(
			self.__computeRate, 
			axis = 1) 
	
	def getAvgRate(self):
		return self.__avgRate
	
	def setAvgRate(self, rate): 
		oldRate = self.__avgRate
		self.__avgRate = rate
		return oldRate

	def __checkRowValidity(self, row, minTime = 0, maxTime = None, maxTimeGap = 600):
		if (maxTime == None) or (maxTime > self.__marketCloseTime): 
			maxTime = self.__marketCloseTime
		
		return not (
			(abs(row['seconds_call'] - row['seconds_put']) > maxTimeGap) 
			or (row['seconds'] > maxTime) 
			or (row['seconds_call'] < minTime)
			or (row['seconds_put'] < minTime)
    	)

	def __getMaxTime(self, row): 
		return max(row['seconds_call'], row['seconds_put'])
	

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
			
	def __writePairs(self):
		self.__pairs.to_csv('Pairs.csv', index=False)
	
	# Implied Volatility

	def __createInterestRateTable(self, alpha = 0.2): 
		numBuckets = math.floor(self.__marketCloseTime/900)

		raw_rates = []
		for i in range(0, numBuckets): 
			raw_rates.append(self.computeRFRate(900*i, 900*(i+1), 600))

		self.__interest_rates = []
		ema_rate = raw_rates[0] 
		self.__interest_rates.append(ema_rate)
		ema_rate = 0.5 * raw_rates[1] + 0.5 * raw_rates[0] # since the first is bound to be more unstable
		self.__interest_rates.append(ema_rate)
		for r in raw_rates[2:]:
			ema_rate = alpha * r + (1 - alpha) * ema_rate
			self.__interest_rates.append(ema_rate)

		"""plt.plot(raw_rates, label='Raw Rates')
		plt.plot(self.__interest_rates, label='EMA Rates')
		plt.legend()
		plt.show()"""

		print('Interest rates calculated!')

	def computeImpliedVolatilities(self, maxIterations = 100, tolerance = 0.00005): 
		self.df['implied_vol'] = np.nan

		self.__createInterestRateTable()

		self.df['risk_free_rate'] = np.nan

		for idx, row in self.df.iterrows():
			S = row['u_trade_px']
			K = row['strike_price']
			T = row['year_to_expiration']
			t = row['seconds']
			if t>=900: 
				r = self.__interest_rates[math.floor(t/900) - 1]
				self.df.at[idx,'risk_free_rate'] = r
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
		
		print('IV calculations done!')

    # Greeks

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
		
		print('Finished computing Greeks!')

	def __computeDelta(self, row): 
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
		
		vega = 0.01 * S * norm.pdf(d1) * np.sqrt(T)
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
			theta = -1/360 * (S * norm.pdf(d1) * sigma / (2*np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(d2))
		else: 
			theta = -1/360 * (S * norm.pdf(d1) * sigma / (2*np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(-d2))
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
			rho = 0.01 * T * K * np.exp(-r*T) * norm.cdf(d2)
		else: 
			rho = 0.01 * -T * K * np.exp(-r*T) * norm.cdf(-d2)
		return rho


	# Greeks vs Stock Price

	def plotGreeksVsStockPrice(self, symbol, iv_tolerance = 0.03, rate_tolerance = 0.03): 
		option_data = self.df[self.df['symbol'] == symbol]
		
		if option_data.empty:
			print(f"No option found with symbol {symbol}")
			return
		required_columns = ['delta', 'gamma', 'vega', 'theta', 'rho']
		missing = [col for col in required_columns if col not in self.df.columns]
		if missing:
			print(f"Error: Missing Greek columns: {missing}. Run computeGreeks() first.")
			return
		
		mean_r = option_data['risk_free_rate'].mean()
		mean_iv = option_data['implied_vol'].mean()
		K = option_data['strike_price'].iloc[0]

		# Making sure data is comparable
		mask = (
        (option_data['risk_free_rate'] >= mean_r * (1-rate_tolerance)) &
        (option_data['risk_free_rate'] <= mean_r * (1+rate_tolerance)) &
        (option_data['implied_vol'] >= mean_iv * (1-iv_tolerance)) &
        (option_data['implied_vol'] <= mean_iv * (1+iv_tolerance))
		)
		filtered_data = option_data[mask]

		if filtered_data.empty:
			print(f"Risk-free rate and implied volatility aren't consistent enough for {symbol}")
			return

		S = filtered_data['u_trade_px']
		delta = filtered_data['delta']
		gamma = filtered_data['gamma']
		vega = filtered_data['vega']
		theta = filtered_data['theta']
		rho = filtered_data['rho']

		S_min = max(S.min(), K * 0.8)  
		S_max = min(S.max(), K * 1.2)  
		x_padding = (S_max - S_min) * 0.05

		fig, axes = plt.subplots(2, 3, figsize=(15, 10))
		fig.suptitle(f'Greeks vs Stock Price for {symbol} (K={K})', fontsize=16)

		# Delta
		ax = axes[0, 0]
		ax.scatter(S, delta, alpha=0.6, color='blue')
		ax.set_title('Delta vs Stock Price')
		ax.set_xlabel('Stock Price')
		ax.set_ylabel('Delta')
		ax.axvline(K, color='red', linestyle='--', label=f'Strike (K={K})')
		ax.set_xlim(S_min - x_padding, S_max + x_padding)
		ax.grid(True)
		ax.legend()
		
		# Gamma
		ax = axes[0, 1]
		ax.scatter(S, gamma, alpha=0.6, color='green')
		ax.set_title('Gamma vs Stock Price')
		ax.set_xlabel('Stock Price')
		ax.set_ylabel('Gamma')
		ax.axvline(K, color='red', linestyle='--', label=f'Strike (K={K})')
		ax.set_xlim(S_min - x_padding, S_max + x_padding)
		ax.grid(True)
		
		# Vega
		ax = axes[0, 2]
		ax.scatter(S, vega, alpha=0.6, color='red')
		ax.set_title('Vega vs Stock Price')
		ax.set_xlabel('Stock Price')
		ax.set_ylabel('Vega')
		ax.axvline(K, color='red', linestyle='--', label=f'Strike (K={K})')
		ax.set_xlim(S_min - x_padding, S_max + x_padding)
		ax.grid(True)
		
		# Theta
		ax = axes[1, 0]
		ax.scatter(S, theta, alpha=0.6, color='purple')
		ax.set_title('Theta vs Stock Price')
		ax.set_xlabel('Stock Price')
		ax.set_ylabel('Theta')
		ax.axvline(K, color='red', linestyle='--', label=f'Strike (K={K})')
		ax.set_xlim(S_min - x_padding, S_max + x_padding)
		ax.grid(True)
		
		# Rho
		ax = axes[1, 1]
		ax.scatter(S, rho, alpha=0.6, color='orange')
		ax.set_title('Rho vs Stock Price')
		ax.set_xlabel('Stock Price')
		ax.set_ylabel('Rho')
		ax.axvline(K, color='red', linestyle='--', label=f'Strike (K={K})')
		ax.set_xlim(S_min - x_padding, S_max + x_padding)
		ax.grid(True)
		
		# Remove the empty subplot (since we have 5 Greeks)
		fig.delaxes(axes[1, 2])
		
		plt.tight_layout()
		plt.show()
	

	def showAllGraphs(self): 
		unique_symbols = self.df['symbol'].unique()
		for symbol in unique_symbols:
			self.plotGreeksVsStockPrice(symbol)

	# Identifying mispricings

	def identifyMisPricings(self, tolerance = 0.05): 
		self.df['mispriced'] = False

		self.__computeImpliedVolEMA()
		
		for idx, row in self.df.iterrows(): 
			S = row['u_trade_px']
			K = row['strike_price']
			T = row['year_to_expiration']
			r = row['risk_free_rate']
			is_call = (row['is_call'] == 1)
			if pd.isna(row['implied_vol']): 
				continue
			sigma = row['implied_vol']
			sigma_ema = row['implied_vol_EMA']
			if not np.isfinite(sigma): 
				self.df.at[idx, 'mispriced'] = True
				continue
			ratio = (OptionsData.__computeBSP(S, K, T, r, sigma_ema, is_call) - OptionsData.__computeBSP(S, K, T, r, sigma, is_call))/OptionsData.__computeBSP(S, K, T, r, sigma, is_call)
			if abs(ratio) >= tolerance: 
				self.df.at[idx, 'mispriced'] = True

		print("Mispricings identified!")

	def __computeImpliedVolEMA(self, alpha = 0.01): 
		impliedVolEMA = None
		self.df['implied_vol_EMA'] = None

		for idx, row in self.df.iterrows():
			if pd.isna(row['implied_vol']): 
				continue
			elif impliedVolEMA == None: 
				impliedVolEMA = row['implied_vol']
				self.df.at[idx, 'implied_vol_EMA'] = impliedVolEMA
			elif np.isfinite(row['implied_vol']): 
				impliedVolEMA = alpha * row['implied_vol'] + (1-alpha) * impliedVolEMA
				self.df.at[idx, 'implied_vol_EMA'] = impliedVolEMA
			
			
	@classmethod
	def __computeBSP(cls, S, K, T, r, sigma, is_call):
		d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
		d2 = d1 - sigma * np.sqrt(T)

		if is_call: 
			price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
		else: 
			price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

		return price

	def writeDF(self): 
		self.df.to_csv('OptionsData.csv', index=False)




vale = OptionsData('Options Data - data.csv')

vale.computeImpliedVolatilities()
vale.computeGreeks()
vale.identifyMisPricings()
vale.writeDF()
