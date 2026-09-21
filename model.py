import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from sklearn.decomposition import PCA

import joblib

df=pd.read_csv("HDFS_2k.log_structured.csv")
df.shape
df.info()
df.isnull().sum()
df.duplicated().sum()
df.columns
text = df["Content"]

vectorize=TfidfVectorizer(max_features=500)

X=vectorize.fit_transform(text)

X.shape

scaler = StandardScaler(with_mean=False)

X_scaled = scaler.fit_transform(X)

iso = IsolationForest(
    contamination=0.1,
    random_state=42
)

iso.fit(X_scaled)

df["IsolationForest"] = iso.predict(X_scaled)

df["IsolationForest"] = df["IsolationForest"].map({1:0,-1:1})

print(df["IsolationForest"].value_counts())

lof = LocalOutlierFactor(
    contamination=0.1
)

lof_pred = lof.fit_predict(X_scaled)

df["LOF"] = pd.Series(lof_pred).map({1:0,-1:1})

print(df["LOF"].value_counts())

svm = OneClassSVM(
    kernel='rbf',
    gamma='scale',
    nu=0.1
)

svm.fit(X_scaled)

svm_pred = svm.predict(X_scaled)

df["OneClassSVM"] = pd.Series(svm_pred).map({1:0,-1:1})

print(df["OneClassSVM"].value_counts())

comparison = pd.DataFrame({
    "Isolation Forest":[df["IsolationForest"].sum()],
    "LOF":[df["LOF"].sum()],
    "One-Class SVM":[df["OneClassSVM"].sum()]
})

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled.toarray())

plt.figure(figsize=(7,4))

plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=df["IsolationForest"],
    cmap="coolwarm",
    s=15
)

plt.title("Isolation Forest")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.colorbar(label="Anomaly")
plt.show()

plt.figure(figsize=(7,4))

plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=df["LOF"],
    cmap="viridis",
    s=15
)

plt.title("Local Outlier Factor")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.colorbar(label="Anomaly")
plt.show()

plt.figure(figsize=(7,4))

plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=df["OneClassSVM"],
    cmap="plasma",
    s=15
)

plt.title("One-Class SVM")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.colorbar(label="Anomaly")
plt.show()

import os
os.makedirs("models", exist_ok=True)
joblib.dump(iso, "models/isolation_forest.pkl")
joblib.dump(vectorize, "models/tfidf_vectorizer.pkl")
joblib.dump(pca, "models/pca.pkl")