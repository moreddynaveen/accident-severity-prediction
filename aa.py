import joblib
import numpy as np

# Load the saved model
model_path = 'rta_model_deploy3.joblib'
rf = joblib.load(model_path)

# Print model summary
print("Loaded Model:\n", rf)

# Example: New data for prediction (ensure it matches the feature format used during training)
X_new = np.array([[2, 1, 14, 3, 5, 2, 0, 1, 3]])  # This should be a 2D array

# Make a prediction
prediction = rf.predict(X_new)
print("Prediction:", prediction)

# Print model parameters
print("Model Parameters:\n", rf.get_params())
