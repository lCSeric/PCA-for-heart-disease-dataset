import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets



heart_data = pd.read_csv(r"C:\Users\林承劭\Desktop\大學課程\coding\python\Data Science\PCA-for-heart-disease-dataset\heart.csv")

missing_values = heart_data.isnull().sum().sum()
print(f"Missing values: {missing_values}")


categorical_features = heart_data.select_dtypes(include=["object"]).shape[1]
print(f"Categorical features: {categorical_features}")


X = heart_data.drop(columns=['target'])
y = heart_data['target']
target_names = heart_data['target'].unique() # It will return an array containing all the unique values of the Series

print(f"Number of original feature: {X.shape[1]}") # return columns of X

X = X - np.mean(X, axis = 0) # each colunn subtract by mean value to center the data

from sklearn.decomposition import PCA

pca = PCA(n_components = 2) #  retain 2 principal components
X_r = pca.fit(X).transform(X) # the PCA model calculates the principal components, and use transform(X) to transform the data into a new coordinate system.

print(f"Number of new features: {X_r.shape[1]}")

plt.figure()
colors = ['green', 'yellow', 'red']
lw = 2 # set the line width

for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw, label=target_name)

plt.legend(loc = "best", shadow = False, scatterpoints = 1)
plt.title("PCA of Heart Disease dataset dataset")
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score

def logeistic_regression(X_train, X_test, y_train, y_test):
    model = LogisticRegression(max_iter= 1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average= "weighted")
    return acc, f1

n_components_list = [2, 5, 10]
for n_components in n_components_list:
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size= 0.3, random_state= 42)
    acc, f1 = logeistic_regression(X_train, X_test, y_train, y_test)

    print(f"Number of compotents :{n_components}, Accuracy :{acc}, F-1: {f1}")