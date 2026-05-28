import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib
import mlflow
import mlflow.sklearn

# Load dataset
df = pd.read_csv("data/Placement_Data_Full_Class.csv")

# Select useful columns
df = df[['ssc_p', 'hsc_p', 'degree_p', 'etest_p', 'mba_p', 'status']]

# Convert target column
df['status'] = df['status'].map({'Placed':1, 'Not Placed':0})

# Features and target
X = df.drop('status', axis=1)
y = df['status']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create MLflow experiment
mlflow.set_experiment("Placement_Prediction")

with mlflow.start_run():

    # Create model
    model = DecisionTreeClassifier()

    # Train model
    model.fit(X_train, y_train)

    # Prediction
    preds = model.predict(X_test)

    # Accuracy
    acc = accuracy_score(y_test, preds)

    # Log parameters
    mlflow.log_param("model", "DecisionTree")

    # Log metric
    mlflow.log_metric("accuracy", acc)

    # Log model
    mlflow.sklearn.log_model(model, "model")

    # Save model
    joblib.dump(model, "model/model.pkl")

    print("Accuracy:", acc)