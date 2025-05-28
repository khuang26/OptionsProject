import pandas as pd
import numpy as np


# HELPER FUNCTIONS

# Get interest rate for a given pair
def getRate(row): 
	S = row['u_trade_px']
	K = row['strike_price']
	C = row['trade_price_call']
	P = row['trade_price_put']
	T = row['year_to_expiration']

	if T <= 0 or K <= 0 or (S - C + P) <= 0:
		return np.nan

	r = - (np.log((S - C + P) / K)) / T
	return r


# Print valid put-call pairs
def printPairs(): 
	print(pairs[['strike_price', 'year_to_expiration', 'trade_price_call', 'trade_price_put', 'u_trade_px', 'r']].head())


# Get avg. interest rate across all corresponding put-call pairs
def getAvgRate(): 
	return pairs['rate'].mean()

# Write to CSV
def writePairs(): 
	pairs[['strike_price', 'year_to_expiration', 'u_trade_px', 'trade_price_call', 'trade_price_put', 'rate']].to_csv("Pairs.csv", index=False)



# MAIN CODE

df = pd.read_csv('Options Data - data.csv')

calls = df[df['is_call'] == 1]
puts = df[df['is_call'] == 0]

# DataFrame with matching put-call pairs
pairs = pd.merge(
    calls,
    puts,
    on=['strike_price', 'year_to_expiration', 'u_trade_px'], # Columns that need to match
    suffixes=('_call', '_put')
)

pairs['rate'] = pairs.apply(getRate, axis = 1) # new column "rate" created in DataFrame; axis = 1 --> checks each row

for i in pairs.index:
  if pairs.loc[i, 'rate'] < 0:
    pairs.drop(i, inplace = True)


avgRate = getAvgRate()
print(f"The average risk-free rate across all valid put-call pairs is: {avgRate}")

writePairs()


print(df.info())      # Original data memory usage
print(pairs.info())    # Merged data memory usage