import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def linear_regression_task(csv_file="students.csv", predict_hours=5, save_plot="regression_plot.png"):
    
    df = pd.read_csv(csv_file)

    X = df[["Hours_Studied"]]
    y = df["Exam_Score"]

    model = LinearRegression()

    model.fit(X, y)

    predicted_score = model.predict([[predict_hours]])[0]
    print(f"Predicted exam score for studying {predict_hours} hours is: {predicted_score}")

    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    print(f"Mean Squared Error of the model: {mse}")

    plt.scatter(X, y, color="blue", label="Actual Data")
    plt.plot(X, y_pred, color="red", label="Regression Line")
    plt.xlabel("Hours Studied")
    plt.ylabel("Exam Score")
    plt.title("Linear Regression: Exam Score vs Hours Studied")
    plt.legend()
    plt.savefig(save_plot) 
    plt.close()

    return predicted_score, mse

if __name__ == "__main__":
    linear_regression_task()
