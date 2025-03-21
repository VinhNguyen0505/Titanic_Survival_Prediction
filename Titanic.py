import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler

# Load Dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)
pd.set_option('display.max_columns', None)

# Initialize ML Models
log_model = LogisticRegression(max_iter=500, solver='lbfgs', random_state=42)
rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
xgb_model = XGBClassifier(n_estimators=200, max_depth=10, random_state=42)

print(f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns")

# Handle Missing Data
df.fillna({'Age': df['Age'].median()}, inplace=True)
df.drop(columns=['Cabin'], inplace=True)
df.fillna({'Embarked': df['Embarked'].mode()[0]}, inplace=True)

print(df.isnull().sum())

# Encode Categorical Variables
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

# Select Features & Target Variable
X = df[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'SibSp', 'Parch']]
y = df['Survived']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {X_train.shape[0]} rows")
print(f"Test set size: {y_test.shape[0]} rows")

# Normalize the Dataset (For Neural Network)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the Neural Network Model
model = Sequential([
    Dense(16, activation='relu', input_shape=(X_train.shape[1],)),  # Input Layer
    Dense(8, activation='relu'),  # Hidden Layer
    Dense(1, activation='sigmoid')  # Output Layer
])

# Compile the Model (Before Training)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the Neural Network
history = model.fit(X_train_scaled, y_train, epochs=50, batch_size=16, verbose=1)

# Train Traditional ML Models
log_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)

print("Model training completed!")

# Predict on Test Set
y_pred_nn = (model.predict(X_test_scaled) > 0.5).astype("int32")
y_pred_rf = rf_model.predict(X_test)
y_pred_xgb = xgb_model.predict(X_test)

# Evaluate Neural Network Model
loss, accuracy_nn = model.evaluate(X_test_scaled, y_test)
print(f"Neural Network Accuracy: {accuracy_nn:.2f}")

# Confusion Matrix & Classification Report for NN
conf_matrix_nn = confusion_matrix(y_test, y_pred_nn)
print("Neural Network Confusion Matrix:")
print(conf_matrix_nn)
print("Neural Network Classification Report:")
print(classification_report(y_test, y_pred_nn))

# Evaluate Random Forest
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {accuracy_rf:.2f}")

# Evaluate XGBoost
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
print(f"XGBoost Accuracy: {accuracy_xgb:.2f}")

# Confusion Matrix & Classification Report for XGBoost
conf_matrix_xgb = confusion_matrix(y_test, y_pred_xgb)
print("XGBoost Confusion Matrix:")
print(conf_matrix_xgb)
print("XGBoost Classification Report:")
print(classification_report(y_test, y_pred_xgb))
