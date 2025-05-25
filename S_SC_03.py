import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

# -------------------------------
# Step 1: Create Sample Dataset
# -------------------------------
data = {
    'age': [30, 40, 50, 35, 28, 60, 45, 33, 55, 38],
    'job': ['admin.', 'technician', 'blue-collar', 'admin.', 'services', 'retired', 'management', 'student', 'retired', 'technician'],
    'marital': ['married', 'single', 'married', 'single', 'divorced', 'married', 'married', 'single', 'divorced', 'married'],
    'education': ['secondary', 'tertiary', 'primary', 'secondary', 'secondary', 'tertiary', 'tertiary', 'secondary', 'primary', 'tertiary'],
    'balance': [1500, 1200, 300, 500, 100, 2000, 2500, 50, 1800, 1400],
    'contact': ['cellular', 'cellular', 'telephone', 'cellular', 'cellular', 'telephone', 'cellular', 'telephone', 'cellular', 'cellular'],
    'day': [5, 10, 15, 7, 12, 20, 18, 3, 14, 6],
    'duration': [100, 200, 150, 90, 300, 250, 120, 80, 110, 220],
    'campaign': [1, 2, 1, 2, 3, 2, 1, 1, 3, 2],
    'pdays': [999, 999, 6, 999, 999, 2, 999, 999, 4, 999],
    'previous': [0, 0, 1, 0, 0, 2, 0, 0, 1, 0],
    'y': ['no', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'no', 'yes', 'yes']
}
df = pd.DataFrame(data)

# -------------------------------
# Step 2: Preprocessing
# -------------------------------
df_encoded = pd.get_dummies(df.drop('y', axis=1))
df_encoded['y'] = df['y'].map({'no': 0, 'yes': 1})

X = df_encoded.drop('y', axis=1)
y = df_encoded['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# -------------------------------
# Step 3: Train Decision Tree
# -------------------------------
model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)

# -------------------------------
# Step 4: Evaluation
# -------------------------------
y_pred = model.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n🧾 Classification Report:\n", classification_report(y_test, y_pred))

# -------------------------------
# Step 5: Visualize Tree
# -------------------------------
plt.figure(figsize=(16, 10))
plot_tree(model, feature_names=X.columns, class_names=['No', 'Yes'], filled=True)
plt.title("Decision Tree - Predicting Customer Subscription")
plt.show()
