import pandas as pd # for dataframe manipulation
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

# reading data from csv into pandas dataframe
data = pd.read_csv("marketing_campaign.csv", sep=";")

print(data.info(verbose=True))  # to check for null entries

# in "Income column there are 24 null entries"

data = data.dropna()  # since there are only 24 null entries, it is best to drop this data, rather than estimating...
# ...some filler values through a categorical average or regression analysis

print(data.dtypes)  # Dt_Customer is not the right type, should be datetime

# changing type to datetime
data["Dt_Customer"] = pd.to_datetime(data["Dt_Customer"], dayfirst=True)

# changing the Dt_Customer variable into a more comparable format, by transforming it into the number of days joined...
# ...in reference to the newest customer, hence the newest customer would be 0

customer_join_dates = []  # creating a list to store dates as a placeholder, for transformation of the data
for x in data["Dt_Customer"]:
    x = x.date()
    customer_join_dates.append(x)

reference_customer = max(customer_join_dates)

days = []  # transformed Dt_Customer variable

for y in customer_join_dates:
    number_of_days = reference_customer-y
    days.append(number_of_days)

data["Dt_Customer"] = days  # overwrites column
data["Dt_Customer"] = data["Dt_Customer"].dt.days  # changes column to

# the birth year was changed into age in the data cleaning, however converting this into a categorical,...
# ...specifically into generations might yield better results, as the difference of consumption behaviour will be...
# ... more pronounced compared to the difference of a few years

year = []  # working table for conversion of variable

for x in data["Year_Birth"]:
    year.append(x)

# lists for different generations

the_greatest_generation = []
for x in range(min(year), 1927):
    the_greatest_generation.append(x)

the_silent_generation = []
for x in range(1928, 1946):
    the_silent_generation.append(x)

baby_boomers = []
for x in range(1946, 1965):
    baby_boomers.append(x)

generation_x = []
for x in range(1965, 1981):
    generation_x.append(x)

millennials = []
for x in range(1981, 1997):
    millennials.append(x)

generations_list = []

# checking in which generation each person falls

for x in year:
    if x in the_greatest_generation:
        generations_list.append("The Greatest Generation")
    if x in the_silent_generation:
        generations_list.append("The Silent Generation")
    if x in baby_boomers:
        generations_list.append("The Baby Boomer Generation")
    if x in generation_x:
        generations_list.append("Generation X")
    if x in millennials:
        generations_list.append("Millennials")


data["Year_Birth"] = generations_list  # transforming the data

# creating a variable for total expenditure, this
data["Expenditure"] = data["MntWines"] + data["MntFruits"] + data["MntMeatProducts"] + data["MntFishProducts"] + data["MntSweetProducts"] + data["MntGoldProds"]

# creating a variable for Household_size

partner = []  # list to hold information concerning marital status: partner or no partner

for x in data["Marital_Status"]:
    if x == "Married":
        partner.append(1)
    elif x == "Together":
        partner.append(1)
    else:
        partner.append(0)

children = []  # list to hold information regarding children

data["Children"] = data["Kidhome"] + data["Teenhome"]

for x in data["Children"]:
    children.append(x)

household_size = list(map(lambda a, b: a+b+1, partner, children))

data["Household Size"] = household_size

#  changing absurd and yolo, since they probably best represent a person who is single
# the data has not been transformed like in the original data cleaning, since different there may be differences in...
# ...in the marital statuses (the ones that actually exist)
data["Marital_Status"] = data["Marital_Status"].replace({"Absurd": "Single", "YOLO": "Single",})

# dropping Data (useless features):

to_drop = ["Z_CostContact", "Z_Revenue", "ID", "Children"]
data = data.drop(to_drop, axis=1)
#  dropping outliers according to original data cleaning

data = data[(data["Year_Birth"] != "The Greatest Generation")]
data = data[(data["Income"] < 600000)]
data["Education"] = data["Education"].replace({"2n Cycle": "2nd_Cycle"})

categorical_cols = ['Marital_Status', 'Year_Birth']
ordinal_cols = ['Education']
numerical_cols = ['Income', 'Expenditure', 'Household Size', 'Dt_Customer', 'Kidhome', 'Teenhome', 'Recency',
                  'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds',
                  'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases',
                  'NumWebVisitsMonth', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1', 'AcceptedCmp2',
                  'Complain', 'Response']  # Accepted CMp is an already one hot encoded variable, as well as...
# ...Response and Complain

# Create a transformer for each data type
# Transfomer for categorical data based on OHC
categorical_transformer = Pipeline(steps=[
    ("encoder", OneHotEncoder(handle_unknown="ignore", drop="first", sparse=False))
])

# Encoding of ordinal data
ordinal_transformer = Pipeline(steps=[
    ("encoder", OrdinalEncoder(categories=[['Basic','Graduation','2nd_Cycle','Master','PhD']]))
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
plt.title('KMeans Clustering with 5 Clusters')
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
axis.scatter3D(xdata, ydata, zdata, c=clusters, cmap = "rainbow", s=100)
axis.set_xlabel("Principal Component 1")
axis.set_ylabel("Principal Component 2")
axis.set_zlabel("Principal Component 3")
plt.show()