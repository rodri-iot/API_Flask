import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import joblib

X, y = load_iris(return_X_y=True, as_frame=True)
df_raw = X
df_raw['y'] = y
# df_raw.info()

df_train, df_test = train_test_split(df_raw, test_size=0.2, random_state=2024, stratify=df_raw['y'])

clf_tree = DecisionTreeClassifier()
clf_tree.fit(df_train.drop('y', axis=1), df_train['y'])

y_pred = clf_tree.predict(df_test.drop('y', axis=1))
accuracy_score(df_test['y'], y_pred)

try:
    joblib.dump(clf_tree, "my_model.joblib")
except Exception as e:
    print(f"Error: {e}")