# Load the library
# you might have to install this the first time
library(mlbench)

# Load the data
ozone = data(Ozone, package='mlbench')

# Look at the help file for details
?Ozone

# Scrub the missing values
# Extract the relevant columns 
ozone = na.omit(Ozone)[,4:13]

y = ozone[,1]
x = as.matrix(ozone[,2:10])

# add an intercept
x = cbind(1,x)

# compute the estimator
betahat = solve(t(x) %*% x) %*% t(x) %*% y

# Fill in the blank
#First get predicted y values
yhat = x %*% betahat

#Then get sum of squares of residuals
ssr =  sum((yhat-y)^2)

#Then divide by degrees of fredom to get estimate of sigma^2
sigma2_hat = ssr/(length(yhat) - length(betahat))

#Then covariance matrix is this variance times the inverse of xxT per earlier part
betacov = sigma2_hat*solve(t(x) %*% x)

# betacov = ?

# Now compare to lm
# the 'minus 1' notation says not to fit an intercept (we've already hard-coded it as an extra column)
lm1 = lm(y~x-1)

summary(lm1)
betacovlm = vcov(lm1)
sqrt(diag(betacovlm))

#Finally compare the results, which gives matrix of basically all zeros:
betacov - betacovlm

