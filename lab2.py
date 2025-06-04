import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load data from local CSV
data = pd.read_csv("housing.csv")

# Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
plt.title('Feature Correlation Matrix')
plt.show()

# Pairplot
sns.pairplot(data, diag_kind='kde', plot_kws={'alpha': 0.5})
plt.suptitle('Feature Pair Relationships', y=1.02)
plt.show()
