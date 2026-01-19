import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 📥 Load CSV files
patients_df = pd.read_csv('Data/PATIENTS.csv')
admissions_df = pd.read_csv('Data/ADMISSIONS.csv')
records_df = pd.read_csv('Data/structured_medical_records.csv')

# 🔗 Merge PATIENTS and ADMISSIONS on 'subject_id'
merged_df = pd.merge(patients_df, admissions_df, on='subject_id', how='inner')

# 📊 Plotting
plt.figure(figsize=(15, 10))

# 1. Histogram of patient ages
plt.subplot(2, 2, 1)
sns.histplot(patients_df['anchor_age'].dropna(), bins=30, kde=True, color='skyblue')
plt.title('Distribution of Patient Age')
plt.xlabel('Age')
plt.ylabel('Count')

# 2. Count of admission types
plt.subplot(2, 2, 2)
if 'admission_type' in admissions_df.columns:
    sns.countplot(data=admissions_df, x='admission_type',
                  order=admissions_df['admission_type'].value_counts().index,
                  palette='muted')
    plt.title('Admission Types Distribution')
    plt.xticks(rotation=15)
else:
    plt.text(0.5, 0.5, 'admission_type column missing', ha='center')

# 3. Gender Distribution Pie Chart
plt.subplot(2, 2, 3)
if 'gender' in patients_df.columns:
    patients_df['gender'].value_counts().plot.pie(autopct='%1.1f%%', colors=['#66b3ff', '#ff9999'])
    plt.title('Gender Distribution')
    plt.ylabel('')
else:
    plt.text(0.5, 0.5, 'gender column missing', ha='center')

# 4. Correlation heatmap from structured records
plt.subplot(2, 2, 4)
numeric_data = records_df.select_dtypes(include=[np.number])
sns.heatmap(numeric_data.corr(), cmap='coolwarm', center=0)
plt.title('Feature Correlation (Structured Records)')

plt.tight_layout()
plt.savefig('output_visuals.png')
plt.show()

# 🧠 ML Model
if 'SepsisLabel' not in records_df.columns:
    raise ValueError("Missing target column 'SepsisLabel' in structured_medical_records.csv")

X = records_df.drop(columns=['SepsisLabel'])
X = X.select_dtypes(include=[np.number])
y = records_df['SepsisLabel']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 📈 Results
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"\nModel Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", report)

# 💾 Save results
with open('results.txt', 'w') as f:
    f.write(f"Dataset shape (structured_medical_records.csv): {records_df.shape}\n")
    f.write("\nMissing Values:\n" + records_df.isnull().sum().to_string())
    f.write("\n\nModel Accuracy: {:.2f}\n".format(accuracy))
    f.write("\nClassification Report:\n" + report)

print("✅ Done! Visuals saved as 'output_visuals.png' and results in 'results.txt'.")