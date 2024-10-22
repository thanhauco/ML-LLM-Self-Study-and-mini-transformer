import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

class LogisticRegression:
    """
    Binary logistic regression using gradient descent.
    """
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = sigmoid(linear_model)

            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)

if __name__ == "__main__":
    # Test case: Binary classification
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [1, 5], [2, 6], [3, 7]])
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    model = LogisticRegression(learning_rate=0.1, n_iterations=1000)
    model.fit(X, y)

    test_X = np.array([[1, 1], [5, 7]])
    predictions = model.predict(test_X)
    print(f"Predictions for test points: {predictions} (Expected: [0 1])")
