import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from google.colab import files

uploaded = files.upload()

data = pd.read_csv("D3.csv")

data = data.apply(pd.to_numeric, errors='coerce')
data.dropna(inplace=True)

x1, x2, x3, y = data.iloc[:, 0], data.iloc[:, 1], data.iloc[:, 2], data.iloc[:, 3]

def gradient_descent(x, y, learning_rate=0.05, iterations=1000):
    m, b = 0, 0
    n = len(y)
    loss_history = []
    
    for _ in range(iterations):
        y_pred = m * x + b
        loss = np.mean((y_pred - y) ** 2)
        loss_history.append(loss)
        
        dm = (-2/n) * np.dot(x, (y - y_pred))
        db = (-2/n) * np.sum(y - y_pred)
        
        m -= learning_rate * dm
        b -= learning_rate * db
    
    return m, b, loss_history

models = {}
for i, x in enumerate([x1, x2, x3], start=1):
    m, b, loss = gradient_descent(x, y)
    models[f"x{i}"] = (m, b, loss)
    
    plt.plot(loss, label=f"x{i}")
    
    print(f"Model for x{i}: y = {m:.4f}x + {b:.4f}")

plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss Over Iterations for Each Explanatory Variable")
plt.show()

def multivariate_gradient_descent(X, y, learning_rate=0.05, iterations=1000):
    n, num_features = X.shape
    theta = np.zeros(num_features)
    b = 0
    loss_history = []
    
    for _ in range(iterations):
        y_pred = np.dot(X, theta) + b
        loss = np.mean((y_pred - y) ** 2)
        loss_history.append(loss)
        
        dtheta = (-2/n) * np.dot(X.T, (y - y_pred))
        db = (-2/n) * np.sum(y - y_pred)
        
        theta -= learning_rate * dtheta
        b -= learning_rate * db
    
    return theta, b, loss_history

X = data.iloc[:, :3].values
theta, b, loss = multivariate_gradient_descent(X, y)

print(f"Multivariate model: y = {theta[0]:.4f}x1 + {theta[1]:.4f}x2 + {theta[2]:.4f}x3 + {b:.4f}")

plt.plot(loss, label="Multivariate")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Loss Over Iterations for Multivariate Model")
plt.legend()
plt.show()

new_data = np.array([[1, 1, 1], [2, 0, 4], [3, 2, 1]])
predictions = np.dot(new_data, theta) + b

print("Predictions for new data points:")
for i, pred in enumerate(predictions):
    print(f"For input {new_data[i]} -> Predicted y = {pred:.4f}")
