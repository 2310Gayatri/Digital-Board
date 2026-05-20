import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pickle

data_dir = "data"
gestures = ["writing", "wiping"]

X = []
y = []

for label, gesture in enumerate(gestures):
    folder = os.path.join(data_dir, gesture)
    for file in os.listdir(folder):
        img_path = os.path.join(folder, file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (64, 64))
        X.append(img.flatten())
        y.append(label)

X = np.array(X)
y = np.array(y)

print("Data loaded:", X.shape)

# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

print("Training complete ")
print("Test accuracy:", knn.score(X_test, y_test))

# Save model
with open("gesture_model.pkl", "wb") as f:
    pickle.dump(knn, f)

print("Model saved as gesture_model.pkl ")
