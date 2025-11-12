# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 10:06:42 2025

@author: EBUNOLUWASIMI
"""

# Step 1: Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Step 2: Create dataset
data = pd.read_csv(
    r"C:\Users\EBUNOLUWASIMI\Dropbox\GM\Data Analytics\prodigy\task 3\Data\bank additional\bank-additional.csv", sep=";")
df = pd.DataFrame(data)
print("Dataframe: ", df.head())

# Step 3: Preprocessing (Encode categorical variable)
df['job'] = df['job'].map({'unknown': 0, 'unemployed': 1, 'housemaid': 2, 'student': 3, 'admin.': 4, 'management': 5,
                          'entrepreneur': 6, 'blue-collar': 7, 'self-employed': 8, 'retired': 9, 'technician': 10, 'services': 11})
df['marital'] = df['marital'].map({'single': 0, 'married': 1, 'divorced': 2})
df['education'] = df['education'].map({'unknown': 0, 'university.degree': 1, 'professional.course': 2,
                                      'illiterate': 3, 'high.school': 4, 'basic.9y': 5, 'basic.6y': 6, 'basic.4y': 7})
df['default'] = df['default'].map({'no': 0, 'yes': 1, 'unknown': 2})
df['housing'] = df['housing'].map({'no': 0, 'yes': 1, 'unknown': 2})
df['loan'] = df['loan'].map({'no': 0, 'yes': 1, 'unknown': 2})
df['contact'] = df['contact'].map({'telephone': 1, 'cellular': 2})
df['month'] = df['month'].map({'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5,
                              'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12})
df['day_of_week'] = df['day_of_week'].map(
    {'mon': 1, 'tue': 2, 'wed': 3, 'thur': 4, 'fri': 5})
df['poutcome'] = df['poutcome'].map(
    {'nonexistent': 0, 'failure': 1, 'success': 2, 'other': 3})
df['y'] = df['y'].map({'no': 0, 'yes': 1})

# Step 4: Split features and target
X = df[['age', 'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'day_of_week', 'month', 'duration',
        'campaign', 'pdays', 'previous', 'poutcome', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']]
y = df['y']

print('X: \n', X, '\n')
print('y: \n', y, '\n')

# Step 5: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# Step 6: Train Decision Tree
entropy = DecisionTreeClassifier(
    criterion='entropy', max_depth=3, random_state=42)
entropy.fit(X_train, y_train)

gini = DecisionTreeClassifier(
    criterion='gini', max_depth=4, min_samples_split=3)
gini.fit(X_train, y_train)

# Step 7: Make predictions
y_entr = entropy.predict(X_test)
y_gini = gini.predict(X_test)

# Step 8: Evaluate performance
print("========== Entropy Model ===========")
print("Accuracy:", accuracy_score(y_test, y_entr))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_entr))
print("\nClassification Report:\n", classification_report(y_test, y_entr))

print("========== Gini Model ===========")
print("Accuracy:", accuracy_score(y_test, y_gini))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_gini))
print("\nClassification Report:\n", classification_report(y_test, y_gini))

# Step 9: Visualize the tree
plt.figure(figsize=(12, 8))
plt.title(label='Entropy model')
plot_tree(entropy, feature_names=X.columns, class_names=[
          'No', 'Yes'], filled=True, rounded=True)
plt.show()

plt.figure(figsize=(12, 8))
plt.title(label='Gini model')
plot_tree(gini, feature_names=X.columns, class_names=[
          'No', 'Yes'], filled=True, rounded=True)
plt.show()
