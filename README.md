
# Time Series Models

If we think back to our lecture on the bias-variance tradeoff, a perfect model is not possible.  There will always be inexplicable error. In time series modeling, we call that noise.  A timeseries that is completely random is called whitenoise, and is written mathematically as:

$$\Large Y_t =  \epsilon_t$$

We know this data has no true pattern governing its fluctuations (because we coded it with a random function).

Any attempt at a model would be fruitless.  The next point in the series could be any value, completely independent of the previous value.

We will assume that the timeseries data that we are working with is more than just white noise.

# Train Test Split

Let's reimport our chicago gun crime data, and prepare it in the same manner as the last notebook.


We are going to resample to the week level for this notebook.

Train test split for a time series is a little different than what we are used to.  Because chronological order matters, we cannot randomly sample points in our data.  Instead, we cut off a portion of our data at the end, and reserve it as our test set.

We will now set aside our test set, and build our model on the train.

# Random Walk

A next logical step would be to simply predict the next data point with the point previous to it.  

We call this type of time series a random walk, and it is written mathematicall like so.

$$\Large Y_t = Y_{t-1} + \epsilon_t$$

$\epsilon$ represents white noise error. 

$$\Large Y_t - Y_{t-1}=  \epsilon_t$$

Let's bring back our Chicago gun crime data and make a simple random walk model.

For a baseline to compare our later models, lets calculate our RMSE for the random walk

Now, lets plot the residuals.

If we look at the rolling standard deviation of our errors, we can see that the performance of our model varies at different points in time.

That is a result of the trends in our data.

In the previous notebook, we were able to make our series stationary by differencing our data. 

Let's repeat that process here. 

In order to make our life easier, we will use statsmodels to difference our data via the ARIMA class. 

We will break down what ARIMA is shortly, but for now, we will focus on the I, which stands for integrated.  A time series which needs to be differenced is said to be integrated [1](https://people.duke.edu/~rnau/411arim.htm). 

There is an order parameter in ARIMA with three slots: (p, d, q).  d represents our order of differencing, so putting a one there in our model will apply a first order difference.




We can see that the differenced predictions (d=1) are just a random walk

By removing the trend from our data, we assume that our data's mean and variance are constant throughout.  But it is not just white noise.  If it were, our models could do no better than random predictions around the mean.  

Our task now is to find more patterns in the series.  

We will focus on the data points near to the point in question.  We can attempt to find patterns to how much influence previous points in the sequence have. 

If that made you think of regression, great! What we will be doing is assigning weights, like our betas, to previous points.

# The Autoregressive Model (AR)

Our next attempt at a model is the autoregressive model, which is a timeseries regressed on its previous values

### $y_{t} = c + \phi_{1}y_{t-1} + \varepsilon_{t}$

The above formula is a first order autoregressive model (AR1), which finds the best fit weight $\phi$ which, multiplied by the point previous to a point in question, yields the best fit model. 

In our ARIMA model, the p variable of the order (p,d,q) represents the AR term.  For a first order AR model, we put a 1 there.

The ARIMA class comes with a nice summary table.  

But, as you may notice, the output does not include RMSE.

It does include AIC. We briefly touched on AIC with linear regression.  It is a metrics that we used to penalize models for having too many features.  A better model has a lower AIC.

Let's compare the first order autoregressive model to our Random Walk.

Our AIC for the AR(1) model is lower than the random walk, indicating improvement.  

Before abandoning it for AIC, let's just make sure the RMSE is lower as well.

Checks out. RMSE is lower as well.

Autoregression, as we said before, is a regression of a time series on lagged values of itself.  

From the summary, we see the coefficient of the 1st lag:

We come close to reproducing this coefficients with linear regression, with slight differences due to how statsmodels performs the regression. 

We can also factor in more than just the most recent point.
$$\large y_{t} = c + \phi_{1}y_{t-1} + \phi_{2}y_{t-2}+ \varepsilon_{t}$$

We refer to the order of our AR model by the number of lags back we go.  The above formula refers to an AR(2) model.  We put a 2 in the p position of the ARIMA class order

Our AIC improves with more lagged terms.

# Moving Average Model (MA)

The next type of model is based on error.  The idea behind the moving average model is to make a prediciton based on how far off we were the day before.

$$\large Y_t = \mu +\epsilon_t + \theta * \epsilon_{t-1}$$

The moving average model is a pretty cool idea. We make a prediction, see how far off we were, then adjust our next prediction by a factor of how far off our pervious predicion was.

In our ARIMA model, the q term of our order (p,d,q) refers to the MA component. To use one lagged error, we put 1 in the q position.


Let's see if we can reproduce the predictions above

Let's look at the 1st order MA model with a 1st order difference

It performs better than a 1st order AR, but worse than a 2nd order

Just like our AR models, we can lag back as far as we want. Our MA(2) model would use the past two lagged terms:

$$\large Y_t = \mu +\epsilon_t + \theta_{t-1} * \epsilon_{t-1} + \theta_2 * \epsilon_{t-2}$$

and our MA term would be two.

# ARMA

We don't have to limit ourselves to just AR or MA.  We can use both AR terms and MA terms.

for example, an ARMA(2,1) model is given by:

 $$\large Y_t = \mu + \phi_1 Y_{t-1}+\phi_2 Y_{t-2}+ \theta \epsilon_{t-1}+\epsilon_t$$


# ACF and PACF

We have been able to reduce our AIC by chance, adding fairly random p,d,q terms.

We have two tools to help guide us in these decisions: the autocorrelation and partial autocorrelation functions.

## ACF

The autocorrelation plot of our time series is simply a version of the correlation plots we used in linear regression.  Our features this time are prior points in the time series, or the lags. 

We can calculate a specific $\gamma_k$ with:

${\displaystyle \gamma_k = \frac 1 n \sum\limits_{t=1}^{n-k} (y_t - \bar{y})(y_{t+k}-\bar{y})}$

The shaded area of the graph is the convidence interval.  When the autocorrelation drops into the shaded area, that means there is no longer statistically significant correlation between lags. 

The above autocorrelation shows that there is correlation between lags up to about 12 weeks back.  

When Looking at the ACF graph for the original data, we see a strong persistent correlation with higher order lags. This is evidence that we should take a first diefference of the data to remove this autocorrelation.

This makes sense, since we are trying to capture the effect of recent lags in our ARMA models, and with high correlation between distant lags, our models will not come close to the true process.

Some rules of thumb:
  - If the autocorrelation shows positive correlation at the first lag, then try adding an AR term.
    
  - If the autocorrelatuion shows negative correlation at the first lag, try adding MA terms.
    
    

This autocorrelation plot can now be used to get an idea of a potential MA term.  Our differenced series shows negative significant correlation at lag of 1 suggests adding 1 MA term.  There is also a statistically significant 2nd, term, so adding another MA is another possibility.


> If the ACF of the differenced series displays a sharp cutoff and/or the lag-1 autocorrelation is negative--i.e., if the series appears slightly "overdifferenced"--then consider adding an MA term to the model. The lag at which the ACF cuts off is the indicated number of MA terms. [Duke](https://people.duke.edu/~rnau/411arim3.htm#signatures)

The ACF can be used to identify the possible structure of time series data. That can be tricky going forward as there often isn’t a single clear-cut interpretation of a sample autocorrelation function.

## PACF

In general, a partial correlation is a conditional correlation. It is the  amount of correlation between a variable and a lag of itself that is not explained by correlations at all lower-order-lags. The autocorrelation of a time series $Y$ at lag 1 is the coefficient of correlation between $Y_t$ and $Y_{t-1}$, which is presumably also the correlation between $Y_{t-1}$ and $Y_{t-2}$. But if $Y_t$ is correlated with $Y_{t-1}$, and $Y_{t-1}$ is equally correlated with $Y_{t-2}$, then we should also expect to find correlation between $Y_t$ and $Y_{t-2}$. Thus, the correlation at lag 1 "propagates" to lag 2 and presumably to higher-order lags. The partial autocorrelation at lag 2 is therefore the difference between the actual correlation at lag 2 and the expected correlation due to the propagation of correlation at lag 1.



When we run a linear regression on our lags, the coefficients calculated factor in the influence of the other variables.  This reminds us of our autoregressive model.  Since the PACF shows the direct effect of previous lags, it helps us choose AR terms.  If there is a significant positive value at a lag, consider adding an AR term according to the number that you see.

Some rules of thumb: 

    - A sharp drop after lag "k" suggests an AR-K model.
    - A gradual decline suggests an MA.

![alt text](./img/armaguidelines.png)

The plots above suggest that we should try a 1st order differenced MA(1) or MA(2) model on our weekly gun offense data.

This aligns with our AIC scores from above.

# auto_arima

Luckily for us, we have a Python package that will help us determine optimal terms.

According to auto_arima, our optimal model is a first order differenced, AR(1)MA(2) model.

Let's plot our training predictions.

# Test

Now that we have chosen our parameters, let's try our model on the test set.

Our predictions on the test set certainly leave something to be desired.  

Let's take another look at our autocorrelation function of the original series.

Let's increase the lags

There seems to be a wave of correlation at around 50 lags.
What is going on?

![verkempt](https://media.giphy.com/media/l3vRhBz4wCpJ9aEuY/giphy.gif)

# SARIMA

Looks like we may have some other forms of seasonality.  Luckily, we have SARIMA, which stands for Seasonal Auto Regressive Integrated Moving Average.  That is a lot.  The statsmodels package is actually called SARIMAX.  The X stands for exogenous, and we are only dealing with endogenous variables, but we can use SARIMAX as a SARIMA.


A seasonal ARIMA model is classified as an **ARIMA(p,d,q)x(P,D,Q)** model, 

    **p** = number of autoregressive (AR) terms 
    **d** = number of differences 
    **q** = number of moving average (MA) terms
     
    **P** = number of seasonal autoregressive (SAR) terms 
    **D** = number of seasonal differences 
    **Q** = number of seasonal moving average (SMA) terms

Let's try the third from the bottom, ARIMA(1, 1, 1)x(0, 1, 1, 52)12 - AIC:973.5518935855749

# Forecast

Lastly, let's predict into the future.

To do so, we refit to our entire training set.
