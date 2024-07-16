# Predictive Analytics on the Titanic Dataset
*[Link to dataset](https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv)*

The Titanic dataset provides information on the passengers aboard the Titanic.

The goal of this project is to build a predictive model to determine whether a passenger survived or not.

Additionally, we will answer [Question 12](https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/problem12.html) and calculate conditional probabilities, belief distributions, and expectations related to survival.

a. Calculate the conditional probability that a person survives given their sex and passenger-class:
$$P(S= true | G=female,C=1)$$
$$P(S= true | G=female,C=2)$$
$$P(S= true | G=female,C=3)$$
$$P(S= true | G=male,C=1)$$
$$P(S= true | G=male,C=2)$$
$$P(S= true | G=male,C=3)$$

b. What is the probability that a child who is in third class and is 10 years old or younger survives? Since the number of data points that satisfy the condition is small use the "bayesian" approach and represent your probability as a beta distribution. Calculate a belief distribution for:
$$S= true | A≤10,C=3$$

c. You can express your answer as a parameterized distribution.
How much did people pay to be on the ship? Calculate the expectation of fare conditioned on passenger-class:
$$E[X | C=1]$$
$$E[X | C=2]$$
$$E[X | C=3]$$
