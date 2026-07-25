import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Load data
df = pd.read_csv("data/creditcard.csv")

# Split features (X) and target (y)
X = df.drop("Class", axis=1)
y = df["Class"]

# Split into training and test sets
# stratify=y keeps the same fraud/normal ratio in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Train a Random Forest model
# class_weight="balanced" tells the model to pay more attention to the rare fraud class
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save the trained model so the dashboard can use it later
joblib.dump(model, "fraud_model.pkl")
print("\nModel saved as fraud_model.pkl")