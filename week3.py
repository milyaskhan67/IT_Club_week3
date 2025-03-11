import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from sklearn.preprocessing import LabelEncoder

# 1. Load and prepare the dataset (assumes a CSV with a target column 'Heart Disease')
df = pd.read_csv('Heart_Disease_Prediction.csv')
X = df.drop(columns=['Heart Disease'])
y = df['Heart Disease']

# Encode the target variable if it's categorical
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split data into training and testing subsets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# 2. Train a Decision Tree model on the training data
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# 3. Evaluate model performance using accuracy, precision, and recall
y_pred = dt_model.predict(X_test)
print("Baseline Decision Tree Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, pos_label=1))
print("Recall:", recall_score(y_test, y_pred, pos_label=1))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 4. Optimize the Decision Tree by experimenting with hyperparameters
param_grid = {
    'max_depth': [None, 3, 5, 7, 10],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42),
                           param_grid,
                           cv=5,
                           scoring='accuracy',
                           n_jobs=-1)
grid_search.fit(X_train, y_train)

# Retrieve the best model and document its hyperparameters
best_dt_model = grid_search.best_estimator_
print("\nBest Hyperparameters:", grid_search.best_params_)

# Evaluate the optimized model performance
y_pred_optimized = best_dt_model.predict(X_test)
print("\nOptimized Decision Tree Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred_optimized))
print("Precision:", precision_score(y_test, y_pred_optimized, pos_label=1))
print("Recall:", recall_score(y_test, y_pred_optimized, pos_label=1))
print("\nClassification Report:\n", classification_report(y_test, y_pred_optimized))