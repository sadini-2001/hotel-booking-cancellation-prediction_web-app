import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import joblib


df = pd.read_csv("data//hotel_booking_cancellation.csv")

# Convert 'arrival_date_month' to numerical values
months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
df['arrival_date_month'] = df['arrival_date_month'].apply(lambda x: months.index(x) + 1)

# One-hot encode the specified columns
one_hot_cols = ['hotel', 'meal', 'market_segment', 'distribution_channel', 'deposit_type', 'customer_type','continent']
df = pd.get_dummies(df, columns=one_hot_cols, drop_first=False)

X = df.drop('is_canceled', axis=1)
y = df['is_canceled']

# Splitting data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

# manual hyperparameter tuning is done since grid did not give good recall for Random Forest

#defining the model
rf_tuned = RandomForestClassifier(
    n_estimators=250,        # More trees for stability
    criterion='gini',        # 'gini' often works better for imbalanced data
    max_depth=24,            # Slightly shallower to reduce overfitting
    min_samples_split=4,     # Helps generalization
    min_samples_leaf=2,      # Avoids very small leaves
    class_weight={0: 1, 1: 2},  # Adjust class weight to balance recall & precision
    random_state=0,
    n_jobs=-1
)

# Train the model
rf_tuned.fit(X_train, y_train)

# Predictions on train and test sets
y_train_pred = rf_tuned.predict(X_train)
y_test_pred = rf_tuned.predict(X_test)

# Evaluate performance
print("==== Train Data Evaluation ====")
print("Train Accuracy:", accuracy_score(y_train, y_train_pred))
print("Train Classification Report:\n", classification_report(y_train, y_train_pred))

print("\n==== Test Data Evaluation ====")
print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
print("Test Classification Report:\n", classification_report(y_test, y_test_pred))


# Save the tuned model
joblib.dump(rf_tuned, "model/model.pkl")

print("Columns after preprocessing:", X.columns.tolist())
