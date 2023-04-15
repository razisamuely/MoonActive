

<p align="center">
  <img src="https://media.tenor.com/tL1FAdqvdFsAAAAC/coin-master-free-spins-today-free-spins-and-coins-today.gif"  width="300" height="200">
</p>


# MoonActive - Predict Prices and Recommend Prices
The main objective of this project is to achieve the following:

1. Predict future incomes.
2. Recommend offers to maximize future incomes.

## Technical Details:

1. The code for predicting incomes can be found in the file named `Task_1_Prediction.ipynb`.
2. The code for recommending offers can be found in the file named `Task_2_Recommendation.ipynb.`
3. The functionality of the project is separated and can be accessed through `utils.py`.
4. The training data and data for offer prediction are stored in the `data` directory.
5. The trained models are stored in the `model` directory.
6. A summary of the project, its results, and a discussion can be found in the `README.md` file. 
It is worth noting that the project structure and logic are described in detail in this file, 
while the code files contain minimal explanations to keep them simple and clear for testing with unseen data.

# results  and discussion

### Note:
As this project was subject to a time constraint, the summary and project flow were designed to reach a final 
result as quickly as possible, while focusing on key issues and concepts. However, several crucial steps that 
are typically part of a standard data science project are missing here, such as comparing multiple algorithms, 
conducting in-depth exploratory data analysis, and performing feature engineering.

It is important to acknowledge that these steps were omitted intentionally to provide a simplified and 
fast version of each of these processes. For instance, a Pandas profiling report was created for quick 
exploratory data analysis, which can be found in profiling_report.html, and only StandardScaler was used 
for fast preprocessing.

At the end of this `README.md` file, i'll detail list of additional steps that must be taken in future 
research will be provided.





# Task 1 - Income prediction


## Final Results:

**RMSE**

1. RMSE on the training data = 77
2. RMSE on the test data = 103


### Top 3 features 
1. If we assume that exceptional points occur at the same rate in unseen data:
   1. `org_price_usd_preceding_30_days`
   2. `tournament_spins_reward_7_preceding`
   3. `org_price_usd_preceding_3_to_7_days`
   
2. If we assume that exceptional points do not represent the majority of future data:
    1. `payment_occurrences_preceding_30_days`
    2. `org_price_usd_preceding_3_days`
    3. `org_price_usd_triple_preceding_30_days` 
   
### Assignment Structure:
1. Split the given data to train test in a ratios of 0.8 , 0.2 respectively 
2. Constructing sklearn pipeline wth the following steps :
   1. Data standardization 
   2. Defining params for turning on XGBregressor  
   3. Cross validating XBRregressor with 5 folds 
   4. Choose model with lower rmse 
3. Repeat **2** but with `RMSLE` loss for testing the effect of the exceptional points on the *top best features*


1. The given data was split into training and testing sets in a ratio of 0.8:0.2.
2. An sklearn pipeline was constructed with the following steps:
   1. Data standardization.
   2. Defining parameters for turning on XGBregressor.
   3. Cross-validating XBRregressor with 5 folds.
   4. Choosing the model with the lower RMSE.
3. Step 2 was repeated, but with RMSLE loss, to test the effect of exceptional points on the top 3 features.

### Discussion
**Results**
**Learning curve**:

As it shown there is a convergence.

**loss**
The loss is defined as `squarederror`, which tends to overemphasize errors in samples with high target values. 
Therefore, feature importance is highly affected by the target, as the model tends to split on features that 
minimize the loss.
Another important point is that it is clear that we have reached a point of overfitting, 
both based on the plot and the difference between the test and train data that was examined above. 
For future research, we will also implement early stopping.

<p align="center">
  <img src="https://github.com/razisamuely/MoonActive/blob/main/gif/loss_convergence.png" width="300" height="200">
</p>

When the same flow is run with the `squaredlogerror` loss, the model looks at the ratio of 
error instead of the euclidean distance. In such a case, we can see a significant difference in feature importance.

**Features**
<p align="center">
  <img src="https://raw.githubusercontent.com/razisamuely/MoonActive/main/gif/featue%20inportance%20RMSE.png?token=GHSAT0AAAAAAB6GIEH4C2T433ITTRWAHIJIZB3CFKQ"  width="300" height="200">
</p>


If we want to minimize the mean squared error (MSE) using the squared error loss, 
we need to ensure that the exceptional points are also present in future data. 
This is because the model tends to split on features that minimize the MSE, 
and if exceptional points are not present in the future data, the model may not perform well on it.



1. `org_price_usd_preceding_30_days`
2. `spins_reward_preceding_30_days`
3. `org_price_usd_preceding_7_to_30_days`


These features suffer from high correlation, as shown in the gif in the top left corner. 
High correlation between features can lead to multicollinearity, which can affect the model's 
stability and interpretability.

<p align="center">
  <img src="https://raw.githubusercontent.com/razisamuely/MoonActive/main/gif/correlation.gif?token=GHSAT0AAAAAAB6GIEH4DNWLVNYYXZMGFHG2ZB3CFWA"  width="300" height="200">
</p>



Especially features 1 and 2 suffer from extreme high correlation, 
with a correlation coefficient of 0.98. In such cases, it is often useful to drop 
one of the highly correlated features to avoid redundancy and improve the model's performance. 
Therefore, we could consider dropping feature 2,3 and choosing feature 4,5 instead.


However, before making any final decisions on feature selection, it is important to conduct experiments 
to validate this assumption. We could try running the model with and without feature 3,4 and compare their 
performance. 

Based on the discussion it seems that the top 3 features with low correlation are:

1. `org_price_usd_preceding_30_days`
2. `tournament_spins_reward_7_preceding`
3. `org_price_usd_preceding_3_to_7_days`

If we want to train a model that is less affected by exceptional points but still takes them into account, 
we can use the RMSLE loss. By doing so, we get a different feature importance ranking 
that leads to the following top 3 features:

<p align="center">
  <img src="https://raw.githubusercontent.com/razisamuely/MoonActive/main/gif/featue%20inportance%20RMSLE.png?token=GHSAT0AAAAAAB6GIEH45Z45ZVICFCZ3PMFMZB3CGAA" width="300" height="200">
</p>

 1. `payment_occurrences_preceding_30_days`
 2. `org_price_usd_preceding_3_days`
 3. `org_price_usd_triple_preceding_30_days` 


**data split** 
A 0.8/0.2 train/test split is a common standard in machine learning for evaluating model performance. 
However, it is important to note that the choice of split ratio depends on the size and complexity of the dataset, 
as well as the specific requirements of the task at hand.
In some cases, it may be necessary to use alternative techniques such as cross-validation or time-series splitting 
to ensure that the model is evaluated on a representative sample of the data. In any case, the choice of data 
split method should be based on careful analysis and experimentation to ensure that the model is robust and 
generalizable to new data.

**Algorithm** 

I decided to use the XGBRegressor[XGBregressor](https://xgboost.readthedocs.io/en/stable/python/python_api.html)
algorithm for this project, as it is widely recognized as a reliable and 
powerful tool for analyzing tabular data. Due to time constraints, I did not have the opportunity to 
compare multiple algorithms, but I believe that XGBRegressor was a good choice for this particular project.

# Task 2 - Recommendation
<p align="center">
    <img src="https://media.tenor.com/4ZgDQzw4lg4AAAAM/value-added-check.gif"  width="100" height="60">
</p>


### Final results

**Recommendation vector**
The optimal treatment can be found in the file named `data/optimal_treatment.json`

**Values assignment distribution**
In 70% of the data (140k samples), our model is agnostic to the treatment value, and in these cases, 
we will prefer to assign a treatment value of 10.

In the remaining 30% of the data, our model significantly tends to assign a treatment 
value of 2 (29% of the data) rather than 10.


### Top 3 features 

1. `tournament_spins_reward_7_preceding`
2. `spins_reward_preceding_30_days`
3. `org_price_usd_preceding_3_to_7_days`

<p align="center">
  <img src="https://raw.githubusercontent.com/razisamuely/MoonActive/main/gif/ass_2_diffs.png?token=GHSAT0AAAAAAB6GIEH42RXTA247VAK5FYU4ZB3CGLA"  width="300" height="200">
</p>

Since 70% of the data (140k samples) shows that the model is agnostic to the treatment value, 
I am interested in isolating only the samples where the predicted price differs between 
the two treatments. I calculated the difference between the means of each 
feature after standardization with respect to the all population.
then i selected the features with the highest difference. 
In future research, when encountering such a situation, 
it is highly preferable to conduct a statistical test that takes variance into consideration.


### Assignment Structure:
1. Using only the training set without verifying on the test set:
   1. We are interested in focusing on the process of choosing the best treatment.
   2. The assumption is that our main model and loss are optimal (which should be tested in further research).
   3. given that, we will use all of our training data for the treatment assignment and not split it up for evaluation.
   
2. Constructing sklearn pipeline with the following steps :
   1. Data standardization 
   2. Defining params for turning on XGBregressor  
   3. Cross validating XBRregressor with 5 folds and rmse loss
   
3. To evaluate the model's decisions, we will look at the assignment ratios for treatments 2 and 10
4. We will create a treatment assignment vector for the test set, which will allow for further investigation
5. As previously stated, the method of assignment is as follows: for each data point, 
we will make two predictions - one with treatment = 2 and another with treatment = 10
7. We will then select the treatment with the higher predicted price as the assigned treatment for that data point.
   
## Farther research 
Please note that this is a first and short iteration, and its main goal is to prove the concept. Given this, there is a lot of room for improvement and further research. Some things that could be added include:

Using addtional feature selection process, such as SHAP, regularization penalties, and others
Bigger Grid search to optimize model parameters
Comparison with other ml models such as regression, loightGBM etc
Further EDA to uncover more insights from the data
Deep dive into the differences between the results of the test and train sets
Testing the statistics of the results, such as the mean, max errors and median
Data enrichment by: Featuer creation and features from external sources
Plotting the error against time
Adding a histogram of errors for better visualization.

## Cdoe structur 
Please note that the code for this project is not written in classes as is typically done. 
Instead, the code is organized in a procedural manner for simplicity and ease of understanding.