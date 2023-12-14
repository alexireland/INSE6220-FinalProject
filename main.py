import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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
        return 1
    elif 25 <= val <= 30:
        return 2
    else:
        return 3

df['Class'] = df['BMI'].apply(categorize_value)


print("Dataset has been modified and is ready for analysis.")

# Box plot of the data in column 16 (index 15)
print("Creating box plot for column 16...")
plt.figure(figsize=(8, 6))
sns.boxplot(data=df.iloc[:, 15], orient='v')
plt.title('Box Plot of BMI data')
plt.ylabel('Column 16 values')
plt.show()

# Drop the 42nd data point (row 43, index 41)
df = df.drop(index=41)
# For ease of debugging, create the
df.to_csv('temp.csv', index=False)

# Load modified dataset
print("Loading raw dataset...")
df = pd.read_csv('temp.csv')

# Box plot of the data in column 16 (index 15) without the problematic data point
print("Creating box plot for column 16...")
plt.figure(figsize=(8, 6))
sns.boxplot(data=df.iloc[:, 15], orient='v')
plt.title('Box Plot of Column 16')
plt.ylabel('Column 16 values')
plt.show()

# Box plot of the data in column 16 with points overlaid
print("Creating box plot with points for column 16...")
plt.figure(figsize=(8, 6))
sns.boxplot(data=df.iloc[:, 15], orient='v', color='lightblue')
sns.stripplot(data=df.iloc[:, 15], orient='v', color='darkblue', alpha=0.5)
plt.title('Box Plot of Column 16 with Points')
plt.ylabel('Column 16 values')
plt.show()

# Correlation Graph for the first 16 columns
print("Generating correlation graph...")
plt.figure(figsize=(12, 8))
sns.heatmap(df.iloc[:, :16].corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Graph')
plt.show()

# Select the first 12 columns for PCA for the first pass
print("Preparing data for first PCA...")
df_selected = df.iloc[:, :12]
# df_selected = df.iloc[:, 1:12]

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
plt.colorbar(label='Column 17 value')
plt.title('2D Scatter Plot of the first two Principal Components')
plt.show()

# Apply PCA for the third principal component analysis
print("Applying PCA for PC3 analysis...")
pca_3 = PCA(n_components=3)
principal_components_3 = pca_3.fit_transform(scaled_data)

# Scatter Plot for the third principal component and Column 17
print("Creating scatter plot for PC3 and Column 17...")
plt.figure(figsize=(12, 8))
plt.scatter(principal_components_3[:, 2], range(len(principal_components_3)), c=df.iloc[:, 16], cmap='viridis', alpha=0.7)
plt.xlabel('Principal Component 3')
plt.ylabel('Index')
plt.colorbar(label='Column 17 value')
plt.title('Scatter Plot of PC3 with Colorization by Column 17')
plt.show()

# Select the first 12 columns for PCA for the SECOND pass (without age)
print("Preparing data for first PCA...")
# df_selected = df.iloc[:, :12]
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
plt.colorbar(label='Column 17 value')
plt.title('2D Scatter Plot of the first two Principal Components')
plt.show()

# Apply PCA for the third principal component analysis
print("Applying PCA for PC3 analysis...")
pca_3 = PCA(n_components=3)
principal_components_3 = pca_3.fit_transform(scaled_data)

# Scatter Plot for the third principal component and Column 17
print("Creating scatter plot for PC3 and Column 17...")
plt.figure(figsize=(12, 8))
plt.scatter(principal_components_3[:, 2], range(len(principal_components_3)), c=df.iloc[:, 16], cmap='viridis', alpha=0.7)
plt.xlabel('Principal Component 3')
plt.ylabel('Index')
plt.colorbar(label='Column 17 value')
plt.title('Scatter Plot of PC3 with Colorization by Column 17')
plt.show()

# Save the first two principal components and Column 17 to a CSV file
print("Saving the first two principal components and the classification column to CSV file...")
pca_df = pd.DataFrame(principal_components_2d, columns=['PC1', 'PC2'])
pca_df['class'] = df.iloc[:, 16]  # Add Column 17
pca_df.to_csv('PCA_output.csv', index=False)
print("The first two principal components and the classification column saved to PCA_output.csv")

# Save the first two principal components and Column 17 to a CSV file
print("Saving the first two principal components and the classification column to CSV file...")
pca_df = pd.DataFrame(principal_components_3, columns=['PC1', 'PC2', 'PC2'])
pca_df['class'] = df.iloc[:, 16]  # Add Column 17
pca_df.to_csv('PCA_output3.csv', index=False)
print("The first two principal components and the classification column saved to PCA_output.csv")