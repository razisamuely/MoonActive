import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
import time
from typing import List
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
pd.options.mode.chained_assignment = None

def train_and_tune_xgboost_regressor(X_train: pd, y_train: pd, numeric_cols: List[str], loss, verbose=4):
    # Define the preprocessing steps for numerical features
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])

    # Define the column transformer to apply the appropriate preprocessing steps to each column
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols)])

    # Define the full pipeline by chaining together the preprocessor and XGBRegressor
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', XGBRegressor())])

    # Define the parameter grid to search over
    param_grid = {

        'regressor__colsample_bytree': [0.5],
        'regressor__objective': [f'reg:{loss}'],
        'regressor__n_estimators': [100, 200],
        'regressor__max_depth': [3, ],
        'regressor__learning_rate': [0.01, ],
        'regressor__reg_alpha': [0.1],
        'regressor__reg_lambda': [0.1],

    }

    # Define the grid search with cross-validation
    grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=5, verbose=verbose)

    # Fit the grid search to the data
    grid_search.fit(X_train, y_train)

    return grid_search


def save_model(model, directory, name, time_format="%Y_%m_%d-%H_%M_%S", file_type="json"):
    strings = time.strftime(time_format)
    model_path = f"{directory}/{name}_{strings}.{file_type}"
    joblib.dump(model, model_path)
    return model_path


def load_model(model_path: str):
    return joblib.load(model_path)


def assign_treatment(df, model, treatment_values=[2, 10], past_col_name="treatment", new_col_name="optimal_treatment"):
    if len(treatment_values) != 2:
        raise Exception("This function support treatment_values of len = 2, future version will suports higher values")

    X_train_assigned_a = df[:]
    X_train_assigned_b = df[:]

    a, b = sorted(treatment_values)

    X_train_assigned_a["treatment"] = a
    X_train_assigned_b["treatment"] = b

    df[f"treatment_{a}_predition"] = model.predict(X_train_assigned_a)
    df[f"treatment_{b}_predition"] = model.predict(X_train_assigned_b)

    bigger_condition = df[f"treatment_{a}_predition"] > df[f"treatment_{b}_predition"]

    df["optimal_treatment"] = np.where(bigger_condition, 2, 10)

    return df


def save_optimal_treatment_as_json(data: pd, col, directory, file_name):
    data[col].to_json(f"{directory}/{file_name}.json",orient='columns')


def dictionry_to_class(dictionary: dict):
    class Dobj(object):
        pass

    class_obj = Dobj()
    class_obj.__dict__ = dictionary
    return class_obj


def plot_feature_importance(feature_importances: List[float], feature_names: List[str], top_n: int):
    # Assume `model` is an XGBRegressor model trained on data with known feature names
    feature_importance = feature_importances
    sorted_idx = np.argsort(feature_importance)[::-1][:top_n]

    # Plot feature importances
    plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx])
    plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
    plt.xlabel('Feature Importance')
    plt.title(f'Feature Importance')
    plt.show()


def plot_correlation_top_features(df: pd, top_n: int, feature_importances: List[float], feature_names: List[str]):
    feature_importance = feature_importances
    sorted_idx = np.argsort(feature_importance)[::-1][:top_n]

    # Compute correlations between all pairs of variables
    corr = df[[feature_names[i] for i in sorted_idx][:top_n]].corr()

    fig, ax = plt.subplots(figsize=(30, 15))

    # Plot heatmap of correlations using seaborn
    sns.heatmap(corr, annot=True, cmap='coolwarm')

    # set the title of the plot
    ax.set_title(f"Correlation Matrix top {top_n} features", fontsize=30)

    plot_size = fig.get_size_inches() * fig.dpi
    fontsize = int(plot_size[0] * 0.01)

    # Set the font size for the x and y axis labels
    ax.tick_params(axis='x', labelsize=fontsize)
    ax.tick_params(axis='y', labelsize=fontsize)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=30)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=30)

    # show the plot
    plt.show()


def get_data(path: str, target_col: str, test_size: float, columns_to_drop: str = None):
    df_train = pd.read_csv(path, index_col=0)
    for i in columns_to_drop:
        if i in df_train.columns:
            df_train = df_train.drop(columns=i)

    if target_col in df_train.columns:

        X, y = df_train.drop(columns=[target_col]), df_train[[target_col]]
    else:
        return df_train

    if test_size > 0:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1)

        return X_train, X_test, y_train, y_test
    else:
        return X, y
