# Risk-free rate

Conceptual Understanding: 
1) Find matching put-call pairs (same strike_price, year_to_expiration, and u_trade_px)
2) For each pair, compute the risk-free interest rate using put-call parity (since all other variables are known)
3) Take the average of all calculated risk-free rates

Things to learn: 
- Pandas library (and associated operations)

Improvements for efficiency/cleanliness: 
- Use a weighted average for RF rate based on trade_qty
- Combine identical put-call pairs by summing the trade quantities



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
