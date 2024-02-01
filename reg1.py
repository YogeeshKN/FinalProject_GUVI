import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

#
# Load the training and testing data
def load_data():
    train_data = pd.read_csv('E:\GUVI Projects\Regression1\p1_train.csv')
    test_data = pd.read_csv('E:\GUVI Projects\Regression1\p1_test.csv')
    return train_data, test_data

# Train a linear regression model
def train_linear_regression(train_data, target_column):
    X = train_data.drop(target_column, axis=1)
    y = train_data[target_column]

    model = LinearRegression()
    model.fit(X, y)

    return model

# Evaluate the model on test data
def evaluate_model(model, test_data):
    target_column = test_data.columns[-1]
    X_test = test_data.drop(target_column, axis=1)
    y_test = test_data[target_column]

    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)

    return mse, mae

# Main function
def main():
    # Load data
    train_data, test_data = load_data()

    # Specify the target column
    target_column = train_data.columns[-1]

    # Train linear regression model
    linear_regression_model = train_linear_regression(train_data, target_column)

    # Evaluate the model
    mse, mae = evaluate_model(linear_regression_model, test_data)

    # Print metrics
    print(f"\nMean Squared Error (MSE): {mse}")
    print(f"Mean Absolute Error (MAE): {mae}")

if __name__ == "__main__":
    main()
