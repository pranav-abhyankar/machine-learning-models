# Solar Panel Efficiency Prediction

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split

# Load data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
sample_submission = pd.read_csv('sample_submission.csv')

# Encode categorical variables
categorical = ['string_id', 'error_code', 'installation_type']
label_encoders = {}
for col in categorical:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])
    label_encoders[col] = le

# Feature columns (exclude id and target)
features = [col for col in train.columns if col not in ['id', 'efficiency']]

# Feature Scaling
scaler = StandardScaler()
train[features] = scaler.fit_transform(train[features])
test[features] = scaler.transform(test[features])

# Split train data for validation
X = train[features]
y = train['efficiency']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = LGBMRegressor(random_state=42)
model.fit(X_train, y_train)

# Validation
val_pred = model.predict(X_val)
rmse = mean_squared_error(y_val, val_pred, squared=False)
score = 100 * (1 - rmse)
print(f"Validation RMSE: {rmse:.4f}")
print(f"Score: {score:.2f}")

# Predict on test set
test_pred = model.predict(test[features])

# Create submission
submission = pd.DataFrame({
    'id': test['id'],
    'efficiency': test_pred
})

submission.to_csv('submission.csv', index=False)
print("submission.csv file has been saved.")
