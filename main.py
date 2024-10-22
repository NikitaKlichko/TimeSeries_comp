import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import warnings
import subprocess
import sys
import pickle
import json
import logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

warnings.filterwarnings("ignore")

catboost = False
if not catboost:
  subprocess.check_call([sys.executable, "-m", "pip", "install", "catboost"])
from catboost import CatBoostClassifier

FIT = True
thresh = 0.4
  
clf_model_path = "./model_clf.pickle"
reg_model_path = "./model_reg.pickle"
data_path = "./Данные v2.xlsx"
reg_save_path = "./forecast_value.json"
clf_save_path = "./forecast_class.json"

train_df = pd.read_excel(data_path, "Бр_дневка - 3 (основной)").sort_values("дата").reset_index(drop=True)
test_df = pd.read_excel(data_path, "Прогноз").sort_values("дата").reset_index(drop=True)

def create_lag_features(data, lag=1):
    df = data.copy()
    for i in range(1, lag + 1):
        df[f'lag_{i}'] = df['выход'].shift(i)
    return df.fillna(0)

def create_new_features(df):
  new_df = df.copy()
  new_df = create_lag_features(df, 10)
  new_df['день'] = new_df['дата'].dt.day
  new_df['месяц'] = new_df['дата'].dt.month
  new_df['год'] = new_df['дата'].dt.year
  new_df['день_недели'] = new_df['дата'].dt.dayofweek
  new_df['выходной'] = (new_df['дата'].dt.dayofweek >= 5) * 1.0
  new_df['квартал'] = new_df['дата'].dt.quarter
  new_df['день_года'] = new_df['дата'].dt.dayofyear
  return new_df

def fit_reg_model(X, y):
  logger.info("Regression model training...")
  model = XGBRegressor(n_estimators=100)
  model.fit(X, y)
  return model

def fit_clf_model(X, y):
  logger.info("Classifier model training...")
  model = CatBoostClassifier(verbose=0)
  model.fit(X, y)
  return model

def predict(model, X_test):
  pred = model.predict(X_test)
  return pred

if __name__ == "__main__":

  df_train_feature = create_new_features(train_df)
  df_test_feature = create_new_features(test_df)

  X_train, y_train_reg = df_train_feature.drop(columns=['выход', 'направление', 'дата']), df_train_feature['выход']
  y_train_clf = df_train_feature['направление'].apply(lambda x: 0 if x == 'ш' else 1)
  X_test = df_test_feature.drop(columns=['выход', 'направление', 'дата'])

  if FIT:
    reg_model = fit_reg_model(X_train, y_train_reg)
    with open(reg_model_path, 'wb') as handle:
      pickle.dump(reg_model, handle, protocol=pickle.HIGHEST_PROTOCOL)

    clf_model = fit_clf_model(X_train, y_train_clf)
    with open(clf_model_path, 'wb') as handle:
      pickle.dump(clf_model, handle, protocol=pickle.HIGHEST_PROTOCOL)
      
  else:
    with open(reg_model_path, 'rb') as fp:
      reg_model = pickle.load(fp)
    
    with open(clf_model_path, 'rb') as fp:
      clf_model = pickle.load(fp)
  
  pred_reg = reg_model.predict(X_test)
  pred_reg = [float(x) for x in pred_reg]
  
  pred_clf_proba = clf_model.predict_proba(X_test)[:, 1]
  pred_clf = np.where(pred_clf_proba > thresh, 1, 0)
  pred_clf = [int(x) for x in pred_clf]

  with open(reg_save_path, 'w') as file:
    json.dump(pred_reg, file)
  
  with open(clf_save_path, 'w') as file:
    json.dump(pred_clf, file)