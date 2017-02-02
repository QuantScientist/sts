## Logistic regression

Logistic regression is a generalized linear model, with a binominal distribution and logit link function. The outcome $Y$ is either 1 or 0. What we are interested in is the expected values of $Y$, $E(Y)$. In this case, they can also be thought as probability of getting 1, $p$. However, because $p$ is bounded between 0 and 1, it's hard to implement the method similar to what we did for linear regression. So instead of predicting $p$ directly, we predict the log of odds (logit), which takes values from $-\infty$ to $\infty$. Now the function is: $\log(\frac{p}{1-p})=\theta_0 + \theta_1x_1 + \theta_2x_2 + ...$, let's denote the RHS as $x\cdot\theta$. Next the task is to find $\theta$.

In logistic regresion, the cost function is defined as: $J=-\frac{1}{m}\sum_{i=1}^m(y^{(i)}\log(h(x^{(i)}))+(1-y^{(i)})\log(1-h(x^{(i)})))$, where $h(x)=\frac{1}{1+e^{-x\cdot\theta}}$ is the sigmoid function, inverse of logit function. We can use gradient descent to find the optimal $\theta$ that minimizes $J$. So this is basically the process to construct the model. It is actually simpler than you think when you starting coding.

## Model construction in R

Now let's build our logistic regression. First I will define some useful functions. Note `%*%` is the dot product in R.

```{r, message=FALSE, warning=FALSE}
library(ggplot2)
library(dplyr)
#sigmoid function, inverse of logit
sigmoid <- function(z){1/(1+exp(-z))}

#cost function
cost <- function(theta, X, y){
  m <- length(y) # number of training examples

  h <- sigmoid(X%*%theta)
  J <- (t(-y)%*%log(h)-t(1-y)%*%log(1-h))/m
  J
}

#gradient function
grad <- function(theta, X, y){
  m <- length(y) 

  h <- sigmoid(X%*%theta)
  grad <- (t(X)%*%(h - y))/m
  grad
}

## Training with NBA shot log dataset

Now let's train our model with the [NBA shot log dataset](http://junma5.weebly.com/data-blog/fun-with-advanced-nba-stats). Specifically, I am interested in how will shot clock, shot distance and defender distance affect shooting performance. Naively, we would think _more time remaining in shot clock, shorter distance to basket, farther to defender_ will all increase the probability of a field goal. Shortly, we will see whether we can statistically prove that.

```{r, message=FALSE, warning=FALSE, cache=TRUE}
#load the dataset
shot <- read.csv('2014-2015shot.csv', header = T, stringsAsFactors = F)
shot.df <- select(shot, FGM, SHOT_CLOCK, SHOT_DIST, CLOSE_DEF_DIST)
head(shot.df)

shot.X <- shot.df[, -1]
shot.y <- shot.df[, 1]

mod <- logisticReg(shot.X, shot.y)
mod
```
How do we interpret the model? 

* The first number is the intercept. It is the log odds of a FG if all other predictors are 0. Note if log odds is 0, the probality is 0.5. So the negative intercept means <50%.

* The next three numbers are the coefficients for SHOT_CLOCK, SHOT_DIST, CLOSE_DEF_DIST. For every unit increase in the predictor, the coefficient is the change of log odds while holding other predictors to be constant.

* For example, let's look at the last number. While holding others the same, if the defender moves 1 ft farther away, the log odds of that shot will increase by 0.106.

* If the original FG% is 50%, the new FG% will be 52.6% if the defender is 1 ft farther.

Now, look at the signs of the coefficients, we can conclude that increase in SHOT_CLOCK, CLOSE_DEF_DIST and decrease in SHOT_DIST will all have positive impact in FG%.

