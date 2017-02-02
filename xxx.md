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
