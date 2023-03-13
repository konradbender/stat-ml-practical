#%%
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn import model_selection, ensemble, metrics, neighbors, decomposition
from sklearn import inspection, preprocessing, discriminant_analysis, svm
#%%
raw_x_train = pd.read_csv("data/x_train.csv")
raw_y_train = pd.read_csv("data/y_train.csv")

#%%
X_train, X_val, y_train, y_val = model_selection.train_test_split(
    raw_x_train, raw_y_train, test_size=0.25, random_state=1)

#%%
one_nn = neighbors.KNeighborsClassifier(n_neighbors=1)
one_nn.fit(X_train, y_train["class"].ravel())
y_hat = one_nn.predict(X_val)
accuracy = metrics.accuracy_score(y_val["class"].ravel(), y_hat)
print(f"One-nearest-neighbor has accuracy {accuracy}")

#%%
random_forest = ensemble.RandomForestClassifier(verbose=True,
                                                n_jobs=3)
random_forest.fit(X_train, y_train["class"].ravel())
y_hat = random_forest.predict(X_val)
accuracy = metrics.accuracy_score(y_val["class"].ravel(), y_hat)
print(f"Vanilla Random Forest has accuracy {accuracy}")

#%%
# Idea for the tree:

metrics.ConfusionMatrixDisplay.from_estimator(random_forest,
                                              raw_x_train, raw_y_train["class"].ravel())

#%%
pca = decomposition.PCA(n_components=6)
transformed_x_train = pca.fit_transform(X_train)
transformed_x_val = pca.transform(X_val)


qda = discriminant_analysis.QuadraticDiscriminantAnalysis()
qda.fit(transformed_x_train, y_train["class"].ravel())
y_hat = qda.predict(transformed_x_val)
accuracy = metrics.accuracy_score(y_val["class"].ravel(), y_hat)
print(f"QDA on 6 dimensions has accuracy {accuracy}")

#%%
random_forest = ensemble.RandomForestClassifier(verbose=True,
                                                n_jobs=3)
random_forest.fit(transformed_x_train, y_train["class"].ravel())
y_hat = random_forest.predict(transformed_x_val)
accuracy = metrics.accuracy_score(y_val["class"].ravel(), y_hat)
print(f"Random Forest on 6 dimensions has accuracy {accuracy}")

#%%
model = svm.SVC(kernel="poly", verbose=True)
model.fit(X_train, y_train["class"].ravel())
y_hat = model.predict(X_val)
accuracy = metrics.accuracy_score(y_val["class"].ravel(), y_hat)
print(f"Poly SVM on with degree 3 has accuracy {accuracy}")
#%%
