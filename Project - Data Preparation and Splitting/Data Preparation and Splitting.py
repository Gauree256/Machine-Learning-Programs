import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

# Step 1: Load Dataset
df = pd.read_csv("employees.csv")

print("First 5 rows:\n")
print(df.head())

# Step 1.5: Data Cleaning
print("\nMissing values before cleaning:")
print(df.isna().sum())

print("\nDuplicate rows before cleaning:", df.duplicated().sum())

# Remove duplicate rows
df = df.drop_duplicates().reset_index(drop=True)

# Fill missing numerical values with median
for col in ["Age", "Salary", "Years_Experience"]:
    df[col] = df[col].fillna(df[col].median())

print("\nMissing values after cleaning:")
print(df.isna().sum())

print("\nShape after cleaning:", df.shape)

# Step 2: Label Encoding - Gender
df["Gender_Label"] = df["Gender"].map({
    "Male": 0,
    "Female": 1
})

print("\nAfter Label Encoding Gender:")
print(df[["Name", "Gender", "Gender_Label"]].head())

# Step 3: One-Hot Encoding - Department
df_onehot = pd.get_dummies(
    df,
    columns=["Department"],
    dtype=int
)

print("\nAfter One-Hot Encoding Department:")
print(df_onehot.head())

# Step 4: Features and Label
# Remove target column from features
X = df_onehot.drop("Promotion_Eligibility", axis=1)

# Remove Name because it is not useful for ML
X = X.drop("Name", axis=1)

# Remove original Gender column because Gender_Label is used
X = X.drop("Gender", axis=1)

# Target / Label
y = df_onehot["Promotion_Eligibility"]

print("\nFeatures:")
print(X.head())

print("\nLabel:")
print(y.head())

# Step 5: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTrain shape:", X_train.shape, y_train.shape)
print("Test shape:", X_test.shape, y_test.shape)

# Step 6: Feature Scaling
# Select numerical columns
numeric_columns = X_train.select_dtypes(include="number").columns

# 6.1 Normalization using MinMaxScaler
minmax = MinMaxScaler()

# Fit only on training data
X_train_normalized = X_train.copy()
X_test_normalized = X_test.copy()

X_train_normalized[numeric_columns] = minmax.fit_transform(
    X_train[numeric_columns]
)

X_test_normalized[numeric_columns] = minmax.transform(
    X_test[numeric_columns]
)

print("\nAfter Normalization - First 5 Training Rows:")
print(X_train_normalized.head())

# 6.2 Standardization using StandardScaler
scaler = StandardScaler()

X_train_standardized = X_train.copy()
X_test_standardized = X_test.copy()

# Fit only on training data
X_train_standardized[numeric_columns] = scaler.fit_transform(
    X_train[numeric_columns]
)

X_test_standardized[numeric_columns] = scaler.transform(
    X_test[numeric_columns]
)

print("\nAfter Standardization - First 5 Training Rows:")
print(X_train_standardized.head())

# Final Summary
print("\n========================================")
print("DATA PREPARATION COMPLETED SUCCESSFULLY")
print("========================================")

print("\nOriginal/Cleaned Dataset Shape:", df.shape)
print("Features Shape:", X.shape)
print("Training Features Shape:", X_train.shape)
print("Testing Features Shape:", X_test.shape)
print("Training Labels Shape:", y_train.shape)
print("Testing Labels Shape:", y_test.shape)