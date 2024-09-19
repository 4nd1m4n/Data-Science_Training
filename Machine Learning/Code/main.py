# %%
# Imports
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import xgboost
from bayes_opt import BayesianOptimization
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import (AdaBoostRegressor, BaggingRegressor,
                              GradientBoostingRegressor, RandomForestRegressor,
                              StackingRegressor, VotingRegressor)
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeRegressor

# NOTE: referencing here so it does not get cleaned up by auto import cleaning tools
enable_iterative_imputer

# %%
# Data preparation


def load_data():
    return pd.read_csv('Sleep_Efficiency.csv')


def remove_outliers(data_frame):
    # Calculate the IQR for each column in the dataframe
    Q1 = data_frame.quantile(0.25, numeric_only=True)

    Q3 = data_frame.quantile(0.75, numeric_only=True)

    IQR = Q3 - Q1

    # Print the shape of the dataframe before removing the outliers
    print("The shape of the dataframe before removing the outliers is " +
          str(data_frame.shape))

    # Remove the outliers from the dataframe
    data_frame = data_frame[~((data_frame < (Q1 - 1.5 * IQR)) |
                              (data_frame > (Q3 + 1.5 * IQR))).any(axis=1)]

    # Print the shape of the dataframe after removing the outliers
    print("The shape of the dataframe after removing the outliers is " +
          str(data_frame.shape))

    return data_frame


def get_null_entries(data_frame):
    return data_frame.isnull().sum().to_frame('null_count')


def split_input_output(data_frame):
    y = data_frame['Sleep efficiency']

    # drop id column as it has no learnable information in this context
    # drop sleep efficiency as it is the target value
    X = data_frame.drop(columns=['ID',
                                 'Sleep efficiency'],
                        axis=1)

    return (X, y)


def ordinal_encode(X, ordinal_encoder):
    return pd.DataFrame(ordinal_encoder.fit_transform(X),
                        columns=X.columns)


def impute_null_entries(data_frame):
    imputer_mean = IterativeImputer(missing_values=np.nan,
                                    initial_strategy='mean',
                                    random_state=42)
    return pd.DataFrame(imputer_mean.fit_transform(data_frame),
                        columns=data_frame.columns)


def scale_inputs(X, scaler):
    # Compute the mean and standard deviation of the training set then transform it
    return pd.DataFrame(scaler.fit_transform(X),
                        columns=X.columns)


def clean_data(data_frame, ordinal_encoder, scaler):
    print(f'>> Duplicate entries count:\n\n{data_frame.duplicated().sum()}\n')

    (X, y) = split_input_output(data_frame)

    print(
        f'>> Null entries of input columns:\n\n{get_null_entries(X)}\n')

    X = ordinal_encode(X, ordinal_encoder)
    X = impute_null_entries(X)
    print(
        f'>> Null entries of input columns after label encoding and imputation:\n\n{get_null_entries(X)}\n')

    X = scale_inputs(X, scaler)

    return (X, y)


def histograms(data_frame):
    print(">> Histograms of dataset's columns:")
    data_frame.hist(figsize=(10, 10))
    plt.show()


def heatmap(data_frame):
    print(">> Heatmap of dataset's columns correlation:")
    plt.figure(figsize=(10, 10))
    sns.heatmap(data=data_frame.corr(),
                vmin=-1,
                vmax=1,
                annot=True,
                cmap='coolwarm')


df = load_data()

# NOTE: Commented out because it made the predictions significantly worse
# df = remove_outliers(df)

ordinal_encoder = OrdinalEncoder()
scaler_linear = StandardScaler()
# TODO: CLEAN DATA CANNOT be done on both the training and test data!!! Information Leakage!
# The cleaners/scalers and imputers can only be trained on training but must only be fitted on the test data after that!
(X, y) = clean_data(df, ordinal_encoder, scaler_linear)

histograms(df)

X_and_y = X.copy()
X_and_y['Sleep efficiency'] = y
heatmap(pd.DataFrame.from_dict(X_and_y))

# SO FIRST TRAIN TEST SPLIT - then cleaning and imputing
(X_train, X_test,
 y_train, y_test) = train_test_split(X,
                                     y,
                                     train_size=0.2,
                                     shuffle=True,
                                     random_state=4)


# %%
# K-fold cross validation of various regression ensembles with full dataset

regressors = [
    ('AdaBoostRegressorRFR', AdaBoostRegressor(estimator=RandomForestRegressor())),
    ('GradientBoostingRegressor', GradientBoostingRegressor()),
    ('RandomForestRegressor', RandomForestRegressor()),
    ('XGBRegressor', xgboost.XGBRegressor()),
    ('AdaBoostRegressorDTR', AdaBoostRegressor(estimator=DecisionTreeRegressor())),
    ('Ridge', RidgeCV()),
    ('KNeighborsRegressor', KNeighborsRegressor()),
    ('DecisionTreeRegressor', DecisionTreeRegressor()),
    ('StackingRegressor', StackingRegressor(
        estimators=[('AdaBoostRegressorRFR', AdaBoostRegressor(estimator=RandomForestRegressor())),
                    ('XGBRegressor', xgboost.XGBRegressor())],
        final_estimator=VotingRegressor(
            estimators=[
                ('rf', RandomForestRegressor()),
                ('gbrt', GradientBoostingRegressor())]))),
    ('VotingRegressor', VotingRegressor(
        estimators=[
            ('abr-rfr', AdaBoostRegressor(estimator=RandomForestRegressor())),
            ('gbr', GradientBoostingRegressor()),
            ('rfr', RandomForestRegressor()),
            ('xgb', xgboost.XGBRegressor()),
            ('abr-dtr', AdaBoostRegressor(estimator=DecisionTreeRegressor()))])),
    ('BaggingRegressor', BaggingRegressor(
        estimator=VotingRegressor(
            estimators=[
                ('abr-rfr', AdaBoostRegressor(estimator=RandomForestRegressor())),
                ('gbr', GradientBoostingRegressor()),
                ('rfr', RandomForestRegressor()),
                ('xgb', xgboost.XGBRegressor()),
                ('abr-dtr', AdaBoostRegressor(estimator=DecisionTreeRegressor()))]))),
]

for name, regressor in regressors:
    score = cross_val_score(regressor,
                            X,
                            y,
                            cv=5,
                            scoring='r2',
                            n_jobs=-1)
    print(f'The mean R-squared score of {name} is: {score.mean()}')
    print(
        f'>>  The train/test score of {name} is: {regressor.fit(X_train, y_train).score(X_test, y_test)}\n')


# %%
# Run Bayesian Optimization

def optimizer(max_depth, max_features, learning_rate, n_estimators, subsample):
    params_gbm = dict()
    params_gbm['max_depth'] = round(max_depth)
    params_gbm['max_features'] = max_features
    params_gbm['learning_rate'] = learning_rate
    params_gbm['n_estimators'] = round(n_estimators)
    params_gbm['subsample'] = subsample

    score = cross_val_score(
        GradientBoostingRegressor(random_state=123, **params_gbm),
        X,
        y,
        cv=5,
        scoring='r2',
        n_jobs=-1).mean()

    return score


params_dict = {
    'max_depth': (1, 20),
    'max_features': (0.5, 1),
    'learning_rate': (0.001, 1),
    'n_estimators': (20, 550),
    'subsample': (0.7, 1)
}

optimization = BayesianOptimization(optimizer,
                                    params_dict,
                                    random_state=111)
optimization.maximize(init_points=50,
                      n_iter=25)

params_optimized = optimization.max['params']
params_optimized['max_depth'] = round(params_optimized['max_depth'])
params_optimized['n_estimators'] = round(params_optimized['n_estimators'])
params_optimized['score'] = optimization.max['target']

print(
    f'>> Best parameters for GradientBoostingRegressor:\n\n{json.dumps(params_optimized, indent=2)}')

# %%
# Initialize gradient boosting regressor model with best found parameters

model_best_gbr = GradientBoostingRegressor(random_state=123,
                                           learning_rate=0.016224867884746044,
                                           n_estimators=264,
                                           subsample=0.989757891175945,
                                           max_depth=4,
                                           max_features=0.9135540990898587)

score_best = model_best_gbr.fit(X_train, y_train).score(X_test, y_test)
print(
    f'>> Best gradient boosting regressor r2 (R squared) score on train-test-split dataset:\n{score_best}\n')

# %%
# Predict single subjects sleep-efficiency

predict_input_high_caffeine = pd.DataFrame.from_dict({
    'Age': [34],
    'Gender': ['Male'],
    'Bedtime': ['2021-02-06 23:00:00'],
    'Wakeup time': ['2021-02-07 08:00:00'],
    'Sleep duration': [9.0],
    'REM sleep percentage': [24],
    'Deep sleep percentage': [67],
    'Light sleep percentage': [52],
    'Awakenings': [1.0],
    'Caffeine consumption': [200.0],
    'Alcohol consumption': [0.0],
    'Smoking status': ['No'],
    'Exercise frequency': [4.0]
})

predict_input_low_caffeine = pd.DataFrame.from_dict({
    'Age': [34],
    'Gender': ['Male'],
    'Bedtime': ['2021-02-06 23:00:00'],
    'Wakeup time': ['2021-02-07 08:00:00'],
    'Sleep duration': [9.0],
    'REM sleep percentage': [24],
    'Deep sleep percentage': [67],
    'Light sleep percentage': [52],
    'Awakenings': [1.0],
    'Caffeine consumption': [0.0],
    'Alcohol consumption': [0.0],
    'Smoking status': ['No'],
    'Exercise frequency': [4.0]
})

predict_input_low_caffeine_no_awaking = pd.DataFrame.from_dict({
    'Age': [34],
    'Gender': ['Male'],
    'Bedtime': ['2021-02-06 23:00:00'],
    'Wakeup time': ['2021-02-07 08:00:00'],
    'Sleep duration': [9.0],
    'REM sleep percentage': [24],
    'Deep sleep percentage': [67],
    'Light sleep percentage': [52],
    'Awakenings': [0.0],
    'Caffeine consumption': [0.0],
    'Alcohol consumption': [0.0],
    'Smoking status': ['No'],
    'Exercise frequency': [4.0]
})


def predict(to_predict):
    print(f'>> Test data for single prediction:\n{to_predict}\n')
    to_predict_transformed = ordinal_encoder.transform(
        to_predict)
    prediction = model_best_gbr.predict(to_predict_transformed)
    print(f'>> Predicted sleep efficiency:\n{prediction}\n')


predict(predict_input_high_caffeine)
predict(predict_input_low_caffeine)
predict(predict_input_low_caffeine_no_awaking)

# %%
# Plot most important features with shapley

shap.initjs()
explainer = shap.TreeExplainer(model_best_gbr)
shap_values = explainer.shap_values(X)

i = 4
shap.force_plot(explainer.expected_value,
                shap_values[i], features=X.iloc[i], feature_names=X.columns)

# %%
shap.summary_plot(shap_values, features=X, feature_names=X.columns)

# %%
shap.summary_plot(shap_values, features=X,
                  feature_names=X.columns, plot_type='bar')


# %%
