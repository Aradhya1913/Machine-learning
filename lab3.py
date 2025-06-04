import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

# Load and transform data
iris = load_iris()
data_2d = PCA(n_components=2).fit_transform(iris.data)

# Plot
plt.figure(figsize=(8, 6))
for i, name in enumerate(iris.target_names):
    plt.scatter(data_2d[iris.target==i, 0], data_2d[iris.target==i, 1], 
                label=name, c=['r','g','b'][i])
plt.title('PCA on Iris Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid()
plt.show()