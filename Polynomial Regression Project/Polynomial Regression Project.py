import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def polynomial_regression_task(csv_file="students.csv", degree=2, predict_hours=5, save_plot="polynomial_regression_plot.png"):
   
    df = pd.read_csv(csv_file)  

    X = df[["Hours_Studied"]]  
    y = df["Exam_Score"]        

    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)

    model = LinearRegression()

    model.fit(X_poly, y) 

    predicted_score = model.predict(poly.transform([[predict_hours]]))[0]
    print(f"Predicted exam score for studying {predict_hours} hours is: {predicted_score}")

    y_pred = model.predict(X_poly)  
    mse = mean_squared_error(y, y_pred)
    print(f"Mean Squared Error of the model: {mse}")

    plt.scatter(X, y, color="blue", label="Actual Data")
 
    X_grid = np.linspace(min(X.values), max(X.values), 100).reshape(-1, 1)
    plt.plot(X_grid, model.predict(poly.transform(X_grid)), color="red", label="Polynomial Regression Curve")
    
    plt.xlabel("Hours Studied")
    plt.ylabel("Exam Score")
    plt.title("Polynomial Regression: Exam Score vs Hours Studied")
    plt.legend()
    plt.savefig(save_plot)
    plt.close()

    return predicted_score, mse

if __name__ == "__main__":
    polynomial_regression_task()

