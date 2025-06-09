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
- Almost all of the data points that produce a high risk-free rate are from the first ~10 minutes of the day, after which the risk-free risk rate stabilizes around 14%.
- The other extreme outliers are caused by the same few trades. Some of the trades that cause the extreme outliers include the ones executed at 6852.827 seconds, 14896.105 seconds, and 14854.519 seconds. 
- In response to the trades above, a flurry of response trades occur together (possibly other traders spotting the error?). 

Improving accuracy: 
- Implementing an exponential moving average
  - Learn what it is
  - Learn how to code it

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

# Greeks Derivations

Conceptual Understanding: 
1) Take derivative w.r.t. each variable to derive the formula for the Greeks for a Black-Scholes call option.
2) Use put-call parity to save time to deriving the formula for put options.
3) Calculations: [PDF](https://drive.google.com/file/d/1CnZQocjsDqbdXk7-iaiu2DVfL0O0fhd2/view?usp=sharing)


# Greeks vs Stock Price

Conceptual Understanding: 
1) Choose an option symbol
2) Filter out a selection of options s.t. the risk-free rate and the implied volatility are close enough
3) Plot greek vs stock price for each of the 5 greeks
4) Greeks vs Stock graph: [Desmos](https://www.desmos.com/calculator/zdl7u2bz7f)
