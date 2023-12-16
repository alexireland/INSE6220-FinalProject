# Alexander Ireland – 40292168
# INSE6220 - Fall 2023
# Special Note: Python code, created in collaboration with ChatGTP **
#
# Body Mass Index Classification using Principal Component Analysis and Machine Learning
#
# The code performed the data analysis generates the plots related to the analysis and PCA plots
# it also outputs a datafile PCA_output.csv that is used as the input to machineLearning.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import probplot


def biplot(scores, loadings, labels=None, colors=None):
    xs = scores[:, 0]
    ys = scores[:, 1]
    n = loadings.shape[0]

    scalex = 1.0 / (xs.max() - xs.min())
    scaley = 1.0 / (ys.max() - ys.min())

    if colors is not None:
        plt.scatter(xs * scalex, ys * scaley, c=colors, cmap='viridis')
    else:
        plt.scatter(xs * scalex, ys * scaley)

    for i in range(n):
        plt.arrow(0, 0, loadings[i, 0], loadings[i, 1], color='r', alpha=0.5)
        if labels is not None:
            plt.text(loadings[i, 0] * 1.15, loadings[i, 1] * 1.15, labels[i], color='g', ha='center', va='center')

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid()

# Load dataset
print("Loading raw dataset...")
df = pd.read_csv('bodyfat.csv')

# Rearrange columns: Move 1st, 2nd, and 4th columns to the end
col1 = df.iloc[:, 0]
col2 = df.iloc[:, 1]
col4 = df.iloc[:, 3]
df.drop(df.columns[[0, 1, 3]], axis=1, inplace=True)
df['Density'] = col1
df['Bodyfat'] = col2
df['Weight'] = col4

# Add a BMI column calculated by the formula (703 * column 15 / (column 2^2))
# Assuming column 15 in the new file is now at index 14 after moving columns
df['BMI'] = 703 * df.iloc[:, 14] / (df.iloc[:, 1] ** 2)

# Add an additional column based on conditional logic
def categorize_value(val):
    if val < 25:
        return 0
    elif 25 <= val <= 30:
        return 1
    else:
        return 2

df['Class'] = df['BMI'].apply(categorize_value)


print("Dataset has been modified and is ready for analysis.")

# Box plot of the data in BMI data (index 15)
print("Creating box plot for BMI data...")
plt.figure(figsize=(8, 6))
sns.boxplot(data=df.iloc[:, 15], orient='v')
plt.title('Box Plot of BMI')
plt.ylabel('BMI')
plt.show()

# Drop the 42nd data point (row 43, index 41) because its height is likely erroneous
df = df.drop(index=41)

# ensure df works properly for hte rest of hte script
df.reset_index(drop=True, inplace=True)

# output the processed data to a file for debugging purposes
# df.to_csv('processed_dataset.csv', index=False)

# Box plot of the data in column 16 (index 15) without the problematic data point
print("Creating box plot for BMI without erronous data...")
plt.figure(figsize=(8, 6))
sns.boxplot(data=df.iloc[:, 15], orient='v')
plt.title('Box Plot of BMI')
plt.ylabel('BMI')
plt.show()

# Box plot of the data in column 16 with points overlaid
print("Creating box plot with points for BMI...")
plt.figure(figsize=(8, 6))
sns.boxplot(data=df.iloc[:, 15], orient='v', color='lightblue')
sns.stripplot(data=df.iloc[:, 15], orient='v', color='darkblue', alpha=0.5)
plt.title('Box Plot of BMI with Data Points')
plt.ylabel('BMI')
plt.show()

# Normal Probability Plot for the BMI column
print("Creating normal probability plot for BMI...")
plt.figure(figsize=(8, 6))
probplot(df['BMI'], dist="norm", plot=plt)  # Replace 'BMI' with the actual column name if different
plt.title('Normal Probability Plot for BMI')
plt.show()

# Correlation Graph for the first 16 columns
print("Generating correlation graph...")
plt.figure(figsize=(12, 8))
sns.heatmap(df.iloc[:, :16].corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Graph')
plt.show()

# Select the wanted columns for PCA
print("Preparing data for  PCA...")
df_selected = df.iloc[:, 1:12]

# Standardize the data
print("Standardizing data...")
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_selected)

# Apply PCA for 2D scatter plot
print("Applying PCA for 2D scatter plot...")
pca_2d = PCA(n_components=2)
principal_components_2d = pca_2d.fit_transform(scaled_data)

# 2D Scatter Plot for the first two principal components
print("Creating 2D scatter plot...")
plt.figure(figsize=(12, 8))
plt.scatter(principal_components_2d[:, 0], principal_components_2d[:, 1], c=df.iloc[:, 16], cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Class')
plt.title('2D Scatter Plot of the first two Principal Components')
plt.show()

# Scree Plot with Variance Explained in Percent
print("Generating scree plot...")
pca_all = PCA(n_components=0.999
              )
principal_components_all = pca_all.fit_transform(scaled_data)
explained_variance_percent = pca_all.explained_variance_ratio_ * 100  # Convert to percent
plt.figure(figsize=(10, 6))
plt.plot(range(1, pca_all.n_components_ + 1), explained_variance_percent, 'o-', linewidth=2, color='blue')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained (%)')
plt.xticks(range(1, pca_all.n_components_ + 1))
plt.grid(True)
plt.show()

# Create a biplot
plt.figure(figsize=(12, 7))
biplot(principal_components_all[:, :2], pca_all.components_, labels=df.columns[1:12])
plt.title('Biplot of the first two Principal Components')
plt.show()



# Save the first two principal components and Column 17 to a CSV file
print("Saving the first two principal components and the classification column to CSV file...")
pca_df = pd.DataFrame(principal_components_2d, columns=['PC1', 'PC2'])
pca_df['class'] = df.iloc[:, 16]  # Add Column 17
pca_df.to_csv('PCA_output.csv', index=False)
print("The first two principal components and the classification column saved to PCA_output.csv")

# Create a biplot with colorized points
plt.figure(figsize=(12, 7))
biplot(principal_components_all[:, :2], pca_all.components_, labels=df.columns[1:12], colors=pca_df['class'])
plt.title('Biplot of the first two Principal Components with Colorized Points')
plt.colorbar(label='Class')
plt.show()
