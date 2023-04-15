# MoonActive - predict prices and recommend prices

The goal of this project is to 
1. Predict future incomes 
2. Recommend offers in order to maximize future incomes 

## Technical Details
1. Code for *income prediction* can be found within `Task_1_Prediction.ipynb`.
2. Code for *offer recommendation*  prediction can be found within `Task_2_Recommendation.ipynb`.
3. The functionality is separated and can be found within `utils.py`.
4. The data for training and offers prediction, can be found within the `data` dir.
5. The trained models, can be found within the `model` dir.
6. Summary, results  and discussion can be found here under the `README.md`, 
It important to mentioned that the project structure and logic is detailed here, while the code files contain minimal explenation 
in order to keep it clean and clear for further test with unseen data.


# results  and discussion

### A note 
Since this project has a time limit, both the summary and the 
project flow have been constructed in a way that aims to approach a 
final result as quickly as possible while focusing on the main issues 
and concepts. However, there are many crucial steps that a standard data 
science project should include that are missing here. 
For example, comparing several algorithms, wider exploratory data analysis, 
and feature engineering.

It's important to note that even though I am aware of these missing steps, 
I aim to examine a simplified and fast version of each of these. 
For instance, I created a *Pandas profiling* report for 
quick exploratory data analysis (which can be found under 
`profiling_report.html`), and I used only StandardScaler for fast preprocessing.

Towards the end of this `README.md` file, I will provide a detailed list of some 
additional steps that must be taken in future research.

# Task 1 - Income prediction

### Final results

**RMSE**
1. RMSE on train = 75
2. RMSE on test  = 75 

### Top 3 features 
1. If we believe that exceptional points would be at the same rate in unseen data 
   2. `org_price_usd_preceding_30_days`
   2. `org_price_usd_preceding_3_days`
   3. `tournament_spins_reward_7_preceding`
   
2. If we believe that the exceptional and not representing most of future data:
    1. `payment_occurrences_preceding_30_days`
    2. `org_price_usd_preceding_3_days`
    3. `chests_reward_preceding_30_days` 
   
### Assignment structure 
1. Split the given data to train test in a ratios of 0.8 , 0.2 respectively 
2. Constructing sklearn pipeline wth the following steps :
   1. Data standardization 
   2. Defining params for turning on XGBregressor  
   3. Cross validating XBRregressor with 5 folds 
   4. Choose model with lower rmse 
3. Repeat **2** but with `RMSLE` loss for testing the effect of the exceptional points on the *top best features*


### Discussion
**Results**
**Learning curve**:

As it shown there is a convergence.

**loss**
The loss is defined to be on `squarederror` which push the model to over attention for errors in samples with high target values.
Given that, features importance is highly effected by the target, cause the model tend to split on these that minimize that loss.
When running the same flow with `squaredlogerror` loss, here the model look at the ratio of error and no not the distance.
In such a case we can see also pretty different feature importance.

**Features**
In case we want to minimize the mse only (by using `squarederror` loss) we need to make sure the 
exceptional points would be show also in a future data. 
Even the highest feature importance is among the foolwing features 

1. `org_price_usd_preceding_30_days`
2. `org_price_usd_preceding_7_to_30_days`
3. `org_price_usd_preceding_3_dayss`

These features are suffer from high correlation (As the gif is shows over the left up corner)

<p align="center">
  <img src="https://raw.githubusercontent.com/razisamuely/MoonActive/main/gif/2023-04-14%2021.29.15.gif?token=GHSAT0AAAAAAB6GIEH5X4F6ERHHWZSYROACZB2YZKA"  width="300" height="200">
</p>

Espcially features 1 and 2 suffer from extreme high correlation (0.98)
so we can give up 2 and choose 4 instead.(Future research should this assumption by conductin simple experiment with and without the  feature.)

So the final top 3 features are which as low correlation:
1. `org_price_usd_preceding_30_days`
2. `org_price_usd_preceding_3_days`
3. `tournament_spins_reward_7_preceding`




**data split** - Since there is no available testing data and no knowledge about the 
time relations between point is decide to stick the standard of 0.8, 0.2 train tst split

**Algorithm** - I chose to work with one main algorithms [XGBregressor](https://xgboost.readthedocs.io/en/stable/python/python_api.html)
which considered as the workhorse of the tabular data. And since i had no time 
to compare different algorithm i believe this is a good choice for one shot flow.


# Task 2 - Recommendation
<p align="center">
    <img src="https://media.tenor.com/4ZgDQzw4lg4AAAAM/value-added-check.gif"  width="100" height="60">
</p>


### Final results

**Recommendation vector**
Can found under `data/optimal_treatment.json`

**Values assignment distribution**
70% out of data (140k sampels) our model is agnostic to the treatment value, so in these cases we will prefer to assign 10.

Over the left 30%, our model is significantly tend to assign 2 (29% out of the data) and not 10.

### Top 3 features 


   
