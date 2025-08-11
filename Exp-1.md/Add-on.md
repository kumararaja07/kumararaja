import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,1,1,0])
clf = Perceptron(tol=1e-3, random_state=0)
clf.fit(X, y)
print("Predictions:", clf.predict(X))
for i in range(len(X)):
    if y[i] == 0:
        plt.scatter(X[i][0], X[i][1], color='red')
    else:
        plt.scatter(X[i][0], X[i][1], color='blue')
x_values = [0, 1]
y_values = -(clf.coef_[0][0] * np.array(x_values) + clf.intercept_) / clf.coef_[0][1]
plt.plot(x_values, y_values)
plt.title('Perceptron Decision Boundary for XOR')
plt.xlabel('X1')
plt.ylabel('X2')
plt.grid(True)
plt.show()

output:
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/04144380-ec7a-490c-8de5-a291adc9b4a8" />

