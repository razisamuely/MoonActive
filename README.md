
# MoonActive - Predict Prices and Recommend Prices
The main objective of this project is to achieve the following:

1. Predict future incomes.
2. Recommend offers to maximize future incomes.

## Technical Details:

1. The code for predicting incomes can be found in the file named `Task_1_Prediction.ipynb`.
2. The code for recommending offers can be found in the file named `Task_2_Recommendation.ipynb.`
3. The functionality of the project is separated and can be accessed through utils.py.
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

In the following section of this `README.md` file, a detailed list of additional steps that must be taken in future 
research will be provided.





# Task 1 - Income prediction


## Final Results:

**RMSE**

1. RMSE on the training data = 75
2. RMSE on the test data = 75


### Top 3 features 
1. If we assume that exceptional points occur at the same rate in unseen data:
   2. `org_price_usd_preceding_30_days`
   2. `org_price_usd_preceding_3_days`
   3. `tournament_spins_reward_7_preceding`
   
2. If we assume that exceptional points do not represent the majority of future data:
    1. `payment_occurrences_preceding_30_days`
    2. `org_price_usd_preceding_3_days`
    3. `chests_reward_preceding_30_days` 
   
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

<p align="center">
  <img src="https://raw.githubusercontent.com/razisamuely/MoonActive/main/gif/featue%20inportance%20RMSE.png?token=GHSAT0AAAAAAB6GIEH4E5TRYMGJWU2VI2E4ZB27QRQ"  width="300" height="200">
</p>


When the same flow is run with the `squaredlogerror` loss, the model looks at the ratio of 
error instead of the distance. In such a case, we can see a significant difference in feature importance.

**Features**
<p align="center">
  <img src="https://raw.githubusercontent.com/razisamuely/MoonActive/main/gif/featue%20inportance%20RMSLE.png?token=GHSAT0AAAAAAB6GIEH423CZLDABM2TR3ZFSZB26I4Q"  width="300" height="200">
</p>


If we want to minimize the mean squared error (MSE) using the squared error loss, 
we need to ensure that the exceptional points are also present in future data. 
This is because the model tends to split on features that minimize the MSE, 
and if exceptional points are not present in the future data, the model may not perform well on it.

Even if the highest feature importance is among the following features: [list of features], 
we cannot solely rely on it to ensure the model's accuracy on unseen data. 
We should also evaluate the model's performance on a validation set and use additional 
techniques such as feature selection, regularization, and ensembling to improve its accuracy.

1. `org_price_usd_preceding_30_days`
2. `org_price_usd_preceding_7_to_30_days`
3. `org_price_usd_preceding_3_dayss`



These features suffer from high correlation, as shown in the gif in the top left corner. 
High correlation between features can lead to multicollinearity, which can affect the model's 
stability and interpretability. In such cases, it is often useful to perform feature selection or 
dimensionality reduction techniques such as principal component analysis (PCA) to remove redundant features 
and improve the model's performance.

<p align="center">
  <img src="https://raw.githubusercontent.com/razisamuely/MoonActive/main/gif/correlation.gif?token=GHSAT0AAAAAAB6GIEH4EAD5TRLM2UEEHQW4ZB27NRQ"  width="300" height="200">
</p>



Especially features 1 and 2 suffer from extreme high correlation, 
with a correlation coefficient of 0.98. In such cases, it is often useful to drop 
one of the highly correlated features to avoid redundancy and improve the model's performance. 
Therefore, we could consider dropping feature 2 and choosing feature 4 instead.

However, before making any final decisions on feature selection, it is important to conduct experiments 
to validate this assumption. We could try running the model with and without feature 2 and compare their 
performance on a validation set. If the performance of the model without feature 2 is similar or better 
than the model with feature 2, then dropping it could be a viable option.

Based on the information you provided earlier, it seems that the top 3 features with low correlation are:
1. `org_price_usd_preceding_30_days`
2. `org_price_usd_preceding_3_days`
3. `tournament_spins_reward_7_preceding`

If we want to train a model that is less affected by exceptional points but still takes them into account, 
we can use the RMSLE loss. By doing so, we get a different feature importance ranking 
that leads to the following top 3 features:

<p align="center">
  <img src="https://raw.githubusercontent.com/razisamuely/MoonActive/main/gif/featue%20inportance%20RMSLE.png?token=GHSAT0AAAAAAB6GIEH423CZLDABM2TR3ZFSZB26I4Q">
</p>




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

### Assignment Structure:
1. Using train only without verifying on test
   1. Here interested in focus on the flow of how to choose the best treatment.
   2. The assumption is that our main model and loss is optimal (Which is something which should be tested in further research)
   3. given that, we will use all of our training data for the treatment assignment and not split it up for evaluation.
   
2. Constructing sklearn pipeline wth the following steps :
   1. Data standardization 
   2. Defining params for turning on XGBregressor  
   3. Cross validating XBRregressor with 5 folds and rme loss
   
3. Evaluating the model decisions -> 2 and 10 assignment ratios 

4. Creating treatment assignment vector on the test set for further investigation 
   
