import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


# Load the dataset
football_data = pd.read_csv("footballData.csv")

# Display the first few rows of the dataset to understand its structure
print(football_data.head())

# Visualization of the distribution of the 'overall' rating
sns.histplot(football_data['overall'], kde=True)
plt.title('Distribution of Overall Ratings')
plt.xlabel('Overall Rating')
plt.ylabel('Frequency')
plt.show()

# Dropping columns with high redundancy or low variance
football_data.drop(['player_url', 'sofifa_id', 'defending_marking'], axis=1, inplace=True)

# Handle missing values for numeric columns
numeric_cols = football_data.select_dtypes(include=[np.number])
imputer = SimpleImputer(strategy='median')
numeric_imputed = imputer.fit_transform(numeric_cols)
football_data[numeric_cols.columns] = numeric_imputed

# Handling categorical columns: fill NaNs and encode
categorical_cols = football_data.select_dtypes(include=['object']).columns
for column in categorical_cols:
    football_data[column].fillna('Missing', inplace=True)  # Fill NaNs with a placeholder
    encoder = LabelEncoder()
    football_data[column] = encoder.fit_transform(football_data[column])

# Ensure there are no NaNs left in the dataset
if football_data.isnull().any().any():
    print("Warning: NaNs still present in the dataset!")
else:
    print("No NaNs present in the dataset.")

# Visualization post-cleaning
# Histograms for age and potential
sns.histplot(football_data['age'], kde=True)
plt.title('Age Distribution of Players')
plt.show()

sns.histplot(football_data['potential'], kde=True)
plt.title('Potential Ratings of Players')
plt.show()

# Scatter plot for age vs overall rating
sns.scatterplot(x='age', y='overall', data=football_data)
plt.title('Age vs Overall Rating')
plt.xlabel('Age')
plt.ylabel('Overall Rating')
plt.show()

# Feature Selection using SelectKBest
selector = SelectKBest(f_classif, k=10)
X = football_data.drop('overall', axis=1)
y = football_data['overall']
X_new = selector.fit_transform(X, y)  # Apply feature selection

# Get the selected features
selected_features = X.columns[selector.get_support()]
print("Selected columns:", selected_features)


X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)



# Calculate correlations
corr_matrix = football_data.corr()

# Plot heatmap of correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Matrix of Football Player Attributes')
plt.show()

# Scatter plot example between 'age' and 'overall' to visualize their correlation
sns.scatterplot(x='age', y='overall', data=football_data)
plt.title('Scatter Plot between Age and Overall Rating')
plt.xlabel('Age')
plt.ylabel('Overall Rating')
plt.show()



# Model performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R-squared Score:", r2)


# Plotting predictions vs actual
plt.figure(figsize=(8, 4))
plt.scatter(y_test, y_pred, alpha=0.3)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
plt.title('Actual vs Predicted')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()





# Split data into features and target
X = football_data.drop('overall', axis=1)
y = football_data['overall']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# K-fold cross-validation setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)
model = LinearRegression()

# Perform cross-validation
cv_results = cross_val_score(model, X_train, y_train, cv=kf, scoring='r2')
print("Cross-validation R-squared scores:", cv_results)
print("Average R-squared:", np.mean(cv_results))

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
print("Linear Regression R-squared:", lr_model.score(X_test, y_test))

# Decision Tree
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)
print("Decision Tree R-squared:", dt_model.score(X_test, y_test))

# Random Forest
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)
print("Random Forest R-squared:", rf_model.score(X_test, y_test))

# Gradient Boosting
gb_model = GradientBoostingRegressor(random_state=42)
gb_model.fit(X_train, y_train)
print("Gradient Boosting R-squared:", gb_model.score(X_test, y_test))
