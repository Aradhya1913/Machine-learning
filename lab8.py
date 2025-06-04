from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt


# Load and split data
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and predict
model = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
acc = model.score(X_test, y_test)
print(f"Model Accuracy: {acc * 100:.2f}%")

# Predict a single sample
pred = model.predict([X_test[0]])[0]
print("Predicted Class for the new sample:", "Benign" if pred else "Malignant")

# Plot tree
plt.figure(figsize=(12, 8))
plot_tree(model, filled=True)
plt.title("Decision Tree - Breast Cancer Dataset")
plt.show()