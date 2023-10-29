import pandas as pd
import numpy as np # linear algebra

# data visualization
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib.cm as cm
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import shap

# sklearn
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, OrdinalEncoder
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, silhouette_samples, accuracy_score, classification_report
from sklearn.decomposition import PCA

from pyod.models.ecod import ECOD
from yellowbrick.cluster import KElbowVisualizer
data = pd.read_csv("Exam_Performance_Data.csv", sep=";")

# checking for null values

print(data.info(verbose=True))  # to check for null entries

# no null entries where found

# checking for duplicates, since no IDs where given in the dataset

duplicates = data.duplicated().sum()

print(duplicates)

# one duplicate has been found

data.drop_duplicates(inplace=True)  # duplicate will be dropped

# test scores will be summarized into a meta score, it might be interesting also to see which score is the most...
# ...impactful

data["Test_Meta_Score"] = data["math score"] + data["reading score"] + data["writing score"]

# test preparation will be converted into a simple binary (dummy) variable 0/1, since its impact can be assumed...
# ...quite clearly; lunch was not changed accordingly, since it is not as clear
data["test preparation course"] = data["test preparation course"].replace({"completed": 1, "none": 0})

# transforming education data for the one hot encoding

data['parental level of education'] = data['parental level of education'].replace({'some high school':'some_high_school','high school': 'high_school','some college':'some_college',"associate's degree":"associate's_degree","bachelor's degree":"bachelor's_degree","master's degree":"master's_degree"})

categorical_cols = ['gender', 'race/ethnicity', 'lunch']
ordinal_cols = ['parental level of education']
numerical_cols = ['test preparation course', 'math score', 'reading score', 'writing score', 'Test_Meta_Score']
# test preperation course is an already one hot encoded variable

# Create a transformer for each data type
# Transfomer for categorical data based on OHC
categorical_transformer = Pipeline(steps=[
    ("encoder", OneHotEncoder(handle_unknown="ignore", drop="first", sparse=False))
])

# Encoding of ordinal data
ordinal_transformer = Pipeline(steps=[
    ("encoder", OrdinalEncoder(categories=[['some_high_school','high_school','some_college',"associate's_degree","bachelor's_degree","master's_degree"]]))
])

# Powertransformer normalises data with the assumtption that data is normaly distributed
numerical_transformer = Pipeline(steps=[
    ("transformer", PowerTransformer())
])

# ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_cols),
        ('ord', ordinal_transformer, ordinal_cols),
        ('num', numerical_transformer, numerical_cols)
    ])

# Full pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

pipeline.fit(data)
transformed_data = pipeline.transform(data)

# create a df from transformed data to have a better understanding of data
transformed_df = pd.DataFrame(transformed_data, columns = pipeline.fit(data).get_feature_names_out().tolist())
print(transformed_df)

# Outliers
detector = ECOD()
detector.fit(transformed_df)
outliers = detector.predict(transformed_df)
transformed_df['outliers'] = outliers

data_no_outliers = transformed_df[transformed_df['outliers']==0].drop(["outliers"], axis = 1)
data_with_outliers = transformed_df.copy().drop(["outliers"], axis = 1)

km = KMeans(init="k-means++", random_state=0, n_init="auto")
visualizer = KElbowVisualizer(km, k=(1, 10))

visualizer.fit(data_no_outliers)
visualizer.show()

km = KMeans(n_clusters=3, init='k-means++', random_state=42, n_init = 10)
clusters = km.fit_predict(data_no_outliers)

print(km.cluster_centers_)

data_no_outliers['cluster'] = clusters

## PCA to reduce dimensions and visualise clusters differentiation
import matplotlib.pyplot as plt

# Reduce data to 2D
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_no_outliers)

# Plot
plt.scatter(data_pca[:, 0], data_pca[:, 1], c=clusters, cmap='rainbow', edgecolor='k', s=100)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('KMeans Clustering with 3 Clusters')
plt.show()

## PCA to reduce dimensions and visualise clusters differentiation
import matplotlib.pyplot as plt

# Reduce data to 3D
pca = PCA(n_components=3)
data_pca = pca.fit_transform(data_no_outliers)

# 3D-Plot
graph = plt.figure()
axis = graph.add_subplot(projection="3d")
xdata = data_pca[:, 0]
ydata = data_pca[:, 1]
zdata = data_pca[:, 2]
axis.scatter3D(xdata, ydata, zdata, c=clusters, cmap = "rainbow", s=10)
axis.set_xlabel("Principal Component 1")
axis.set_ylabel("Principal Component 2")
axis.set_zlabel("Principal Component 3")
plt.show()
