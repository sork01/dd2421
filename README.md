# Support Vector Machines


### 6. Running and Reporting
#### Assignments
> 1.Move the clusters around and change their sizes to make it easier or harder for the classifier to find a decent boundary. Pay attention to when the optimizer (minimize function) is not able to find a solution at all.
> 2. Implement some of the non-linear kernels. you should be able to classify very hard datasets.
> 3. The non-linear kernels have parameters; explore how they influence the decision boundary. Reason about this in terms of the bias-variance trade-off.


#### 1. Move the clusters around
The provided data, generated from the code in the text, is linearly separable. Minimize returns true for this data. The Polynomial kernel and the Radial kernel classify the data correctly as well
<p align="center"><img src="https://github.com/sork01/dd2421/blob/master/pic1.png"></p>

#### 2. Implement the non-linear kernels

By changing the dataset I found some sets where the linear kernel could not classify the data but the polynomial and the radial could
<p align="center"><img src="https://github.com/sork01/dd2421/blob/master/pic2.png"></p>

#### 3. Explore non-linear Kernel Parameters
The `p` parameter for the polynomial kernel changes the polynomial degree which in turn leads to decrease in bias and increase in variance. Increasing `sigma` parameter in the radial kernel gives a smoother curve and increase the bias, thus decreasing the variance.
Decreasing the `sigma` increases variance and decreases bias.

<p align="center"><img src="https://github.com/sork01/dd2421/blob/master/pic.gif"></p>

### 4. Slack Implementation
#### Assignments
> 4. Explore the role of the slack parameter C. What happens for very large/small values?
> 5. Imagine that you are given data that is not easily separable. Whenshould you opt for more slack rather than going for a more complex model (kernel) and vice versa?

`C` is a tolerance for data points that are not correctly classified. 
High `C` value means we want to reduce the number of miss-classified points and a low value allows more miss-classified points but instead gives a smooth boundary.

<p align="center"><img src="https://github.com/sork01/dd2421/blob/master/c.png"></p>


Slack Variables are useful when you want to reduce noise when you have a dataset with a large amount of noise.
However, if you know the data does not contain much noise you'd want to use a more complex model. 
That being said, slack variables are a trade-off on the accuracy of the model

