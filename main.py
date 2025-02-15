import numpy as np
import math
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load datasets
X = np.load("X-5-Caltech.npy")  # Modify if using 68 landmarks
y = np.load("y-5-Caltech.npy")

# Feature extraction using Euclidean distance
def extract_features(X):
    features = []
    for person in X:
        features_k = []
        for i in range(person.shape[0]):
            for j in range(person.shape[0]):
                p1, p2 = person[i], person[j]
                features_k.append(math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2))
        features.append(features_k)
    return np.array(features)

X_features = extract_features(X)
X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42)

# Experimenting with different classifiers
classifiers = {
    "KNN": KNeighborsClassifier(n_neighbors=3),
    "SVM": SVC(kernel='linear', C=1.0),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42)
}

for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy:.2f}")
