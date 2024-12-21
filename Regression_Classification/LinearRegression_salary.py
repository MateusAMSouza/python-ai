import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the dataset
salary = pd.read_csv('ExperienceSalary.csv')

# Check for missing values
salary.isnull().sum()
salary = salary.dropna()

# Visualize the data
plt.scatter(salary['exp(in months)'], salary['salary(in thousands)'], marker='*')
plt.xlabel("Experience (in months)")
plt.ylabel("Salary (in thousands)")
plt.show()

# Split the dataset
x_val = salary['exp(in months)'].values.reshape(-1, 1)
y_val = salary['salary(in thousands)']
x_train, x_test, y_train, y_test = train_test_split(x_val, y_val, test_size=0.3, random_state=42)

# Train the model
my_model = LinearRegression()
my_model.fit(x_train, y_train)

# Evaluate the model
print("R_squared =", my_model.score(x_test, y_test))

# Predict and plot results
test_pred = my_model.predict(x_test)
plt.plot(x_test, y_test, '*', label='Real Values')
plt.plot(x_test, test_pred, '-', label='Predicted Values')
plt.xlabel('Experience (in months)')
plt.ylabel('Salary (in thousands)')
plt.legend()
plt.show()

# Print the equation
print(my_model.intercept_, my_model.coef_, my_model.score(x_test, y_test))
