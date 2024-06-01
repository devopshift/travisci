import numpy as np
from sklearn.linear_model import LinearRegression

def train_model():
    # Generate some example data
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3

    # Create and train the model
    model = LinearRegression().fit(X, y)
    
    # Make a prediction
    prediction = model.predict(np.array([[3, 5]]))
    return prediction

if __name__ == "__main__":
    prediction = train_model()
    print(f"Prediction: {prediction}")
