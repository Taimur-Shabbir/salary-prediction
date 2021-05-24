# Problem exposition, and understanding and Data

This project aims to predict salaries for job postings given a set of features that include but is not limited to Degree Requirement, Industry of Job, Miles From Metropolis and Major Required. Thus, it is a supervised learning, regression task. 

There are 1 million observations in the dataset with 8 pre-existing features. I hypothesise that these variables hold significant predictive potential that can considerably outperform a simple baseline prediction

# Approach and Insights

## Cleaning and EDA

I started by cleaning the data and exploring the most interesting variables. Apart from a very small number of erroneous observations, where the outcome (Salary) was given a value of 0, the data was mostly clean and ready to be investigated. I explored univariate distributions of interval variables, including Salary, Years of Experience Required and Miles From Metropolis in addition to bivariate relationships that included finding out how the offered Salary differed among different Industries and with Years of Experience Required. Weak correlations, both negative and positive, were found for the bivariate investigations. The univariate relationships were mostly normaly distributed with a small degree of skewness.

## Feature Engineering

I mostly concentrated on correctly encoding the 4 categorical variables in the data. Two of these were nominal (Degree, Job Type) and two were ordinal (Industry, Major), so manual encoding to preserve the order of values was used in the former case and dummy encoding was used for the latter. The correlation coefficients for these variables were significantly higher than those of the pre-existing variables, ranging in between 0.38 to 0.6

I did not create any interaction variables because no such combination made intuitive sense

## Models

My approaches here included:

- Linear Regression
- Polynomial Regression
- Linear SVM
- Gradient Boosting Regressor

# Results

The best performing model was Polynomial Regression, which resulted in an MSE of 354 units. This is a 74.1% improvement over the baseline model MSE of 1367.12

In terms of what could be improved upon, more advanced models such as neural networks could be used. More feature engineering could be undertaken, such as creating simple minimum, maximum and average salaries for each value of Industry, Degree and Major.
