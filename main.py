import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Import Dataset
dataset = pd.read_csv("car data.csv")

# Data Preparing

dataset.replace({"Fuel_Type":{"Petrol": 0, "Diesel": 1, "CNG":2}}, inplace=True)
dataset.replace({"Transmission":{"Manual": 0, "Automatic": 1}}, inplace=True)
dataset.replace({"Seller_Type":{"Dealer": 0, "Individual": 1}}, inplace=True)

# Separating Independent and Dependent Variables

X = dataset.drop(["Car_Name","Selling_Price"], axis = 1)
y = dataset["Selling_Price"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 42 )

# Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)


# Model Testing
# Note: R-Squared is recommended to use for evaluating the model performance of the regression models.

training_data_prediction = model.predict(X_train)
r_squared_score = metrics.r2_score(y_train, training_data_prediction)
# 0.880617


# Visualization
plt.scatter(y_train, training_data_prediction)
plt.title("Actual vs Prediction", loc = "left")
plt.grid(True, axis="y")
plt.axhline(y=0.1, color='red', linestyle='-')
plt.xlabel("Actual")
plt.ylabel("Prediction")
plt.show()


test_data_prediction = model.predict(X_test)
r_squared_score = metrics.r2_score(y_test, test_data_prediction)
# 0.831106

plt.scatter(y_test, test_data_prediction)
plt.title("Actual vs Prediction", loc = "left")
plt.grid(True, axis="y")
plt.axhline(y=0.1, color='red', linestyle='-')
plt.xlabel("Training")
plt.ylabel("Prediction")
plt.show()

# plt.plot(y_train, color = "red", label = "Actual Prices", linestyle = "dashed", marker = "x")
# plt.plot(training_data, color = "blue", label = "Prediction Prices", linestyle = "--",  marker = ".")


plt.hist(y_train, alpha = 0.5)
plt.hist(training_data_prediction, alpha = 0.5)
plt.show()
