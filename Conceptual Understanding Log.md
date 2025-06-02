# Risk-free rate

Conceptual Understanding: 
1) Find matching put-call pairs (same strike_price, year_to_expiration, and u_trade_px)
2) For each pair, compute the risk-free interest rate using put-call parity (since all other variables are known)
3) Take the average of all calculated risk-free rates

Things to learn: 
- Pandas library (and associated operations)

Fixing code logic: 
- Filter put-call pairs to be within a certain time frame
- Filter put-call pairs that occur too far apart

Improvements for efficiency/cleanliness: 
- Use a weighted average for RF rate based on trade_qty
- Combine identical put-call pairs by summing the trade quantities

Outliers observations ([original data](https://docs.google.com/spreadsheets/d/1BwWRdstB8Nl51dN8z9olRbkPB8Ps4Gcr2hL_aUru6Sc/edit?usp=sharing)): 
- -1.89% (Rows 1448, 15004) --> occured within a second
- -1.66% (Rows 11079, 11104-11107, 11109-11110, 11112, 11166, 24384) --> small call value
- -0.63% (Rows 1448, 15002, 15008, 15017, 15153, 15159) --> no obvious pattern, but there is a cluster of 3 transactions
- 48.45% (Rows 1782, 15548) --> within first 5 minutes of market opening
- 44.65% (Rows 1782, 15549) --> within first 10 minutes of market opening
- 40.22% (Rows 1782, 15841, 15845) --> 1782 again?

- Almost all of the data points that produce a high risk-free rate are from the first ~10 minutes of the day, after which the risk-free risk rate stabilizes around 14%. 

Possible explanation: 
- All caused by a few mispricings (1448, 1782, etc)
- First may be explained by an algorithm taking advantage of the mispricing


# Implied Volatility

Conceptual Understanding: 
1) Now that the risk-free rate is calculated, use the Black Scholes pricing model to calculate the implied volatility based on the data in 'Options Data - data.csv' (since all other variables are known)
2) Use an approximation method (e.g., Newton-Raphson), since 	$\sigma$ can't be written directly in terms of other variables
3) Take an average of all calculated IV
4) (Potentially: make a graph of implied volatility vs time)

Things to learn: 
- Coding an iterative process like Newton-Raphson
  - Learning how derivatives (math) work in Python
- How to code a cumulative distribution function
- (Potentially: Matplotlib)
