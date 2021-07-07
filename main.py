import streamlit as st
from sklearn import datasets
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

st.title("Various classifiers ")

st.write("""Explore different classifiers""")
arr = list()

dataset_name = st.sidebar.selectbox("Select dataset", ("Iris", "Breast Cancer", "Wine dataset","Choose from your device"))
st.write(dataset_name)
classifier_name = st.sidebar.selectbox("Select classifer",("KNN", "Decision Tree","SVM", "Random Forest"))
global df

def get_datset(dataset_name):
    if dataset_name=="Iris":
        fdata = datasets.load_iris()
    elif dataset_name=="Breast Cancer":
        fdata = datasets.load_breast_cancer()
    elif dataset_name=="Wine dataset":
        fdata = datasets.load_wine()
    elif dataset_name=="Choose from your device":
        uploaded_file = st.sidebar.file_uploader(label="Upload your CSV or Excel file ", type=['csv', 'xlsx', 'xlx'])
        if uploaded_file is not None:
            print("Success")
            try:
                df = pd.read_csv(uploaded_file)
            except Exception as e:
                print("exception")
                df = pd.read_excel(uploaded_file)
        st.write("Uploaded data is as follows")
        st.write(df)
    x = fdata.data
    y = fdata.target
    arr = fdata.target_names
    st.write("Features of ", dataset_name)
    st.write(fdata.feature_names)

    return x,y,fdata

st.button("Run Again")

x,y,fdata = get_datset(dataset_name)
st.write("Shape of dataset:", x.shape)
st.write("No. of classes:", len(np.unique(y)))

def variable_k(classifier_name):
    p = dict()
    if classifier_name=="KNN":
        k = st.sidebar.slider("k", 1, 20)
        p["k"] = k

    elif classifier_name=="Decision Tree":
        splitter = st.sidebar.selectbox('Splitter', ('best', 'random'))
        maximum_depth = st.sidebar.slider("max_depth", 2, 15)
        min_samples_split = st.sidebar.slider("min_samples_split", 2, 5)
        min_samples_leaf = st.sidebar.slider("min_samples_leaf", 2, 5)
        p["maximum_depth"] = maximum_depth
        p["splitter"] = splitter
        p["min_samples_split"] = min_samples_split
        p["min_samples_leaf"] = min_samples_leaf

    elif classifier_name=="SVM":
        C = st.sidebar.slider("C", 0.01, 10.0)
        p["C"] = C

    elif classifier_name=="Random Forest":
        maximum_depth = st.sidebar.slider("max_depth", 2, 15)
        no_estimator = st.sidebar.slider("no of estimators", 1, 100)
        p["maximum_depth"] = maximum_depth
        p["no_estimator"] = no_estimator
    return p

p = variable_k(classifier_name)
test_size = st.sidebar.slider("Test data size", 0.1, 1.0)

#getting classifcation algorithm name
def get_classifier(classifier_name,p):
    if classifier_name=="KNN":
        clf = KNeighborsClassifier(n_neighbors=p["k"])
    elif classifier_name=="Decision Tree":
        clf = tree.DecisionTreeClassifier(splitter=p["splitter"], min_samples_split=p["min_samples_split"], min_samples_leaf=p["min_samples_leaf"])
    elif classifier_name=="SVM":
        clf = SVC(C=p["C"],probability=True)
    elif classifier_name=="Random Forest":
        clf  = RandomForestClassifier(n_estimators=p["no_estimator"], max_depth=p["maximum_depth"],random_state=1)
    return clf
clf = get_classifier(classifier_name,p)
#classification

#Spliting data into training and test dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=1)

clf = clf.fit(x_train, y_train)
y_prediction = clf.predict(x_test)

accuracy = accuracy_score(y_test, y_prediction)
st.write("Accuracy of model: ", accuracy)

st.write("Precision of model: ", precision_score(y_test, y_prediction, average='macro'))
st.write("Classifier used: ", classifier_name)

st.write("Test Data")
st.write(x_test)
st.write("Predicted Output for Test data")
st.write(y_prediction)

#Applying PCA and reducing it to 2 dimension

pca = PCA(2)
proj_x = pca.fit_transform(x)
x1 = proj_x[:, 0]
x2 = proj_x[:, 1]
fig = plt.figure()
st.set_option('deprecation.showPyplotGlobalUse', False)
#The alpha blending value, between 0 (transparent) and 1 (opaque).

plt.scatter(x1, x2, c=y, alpha=0.7, cmap="viridis")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar()
st.pyplot()
st.write("Classification done by:", classifier_name, "algorithm")
st.write("Project Done by : Harshit Grover")

#Author: Harshit Grover
#linkedin: https://www.linkedin.com/in/harshit-grover-410a641a0/


