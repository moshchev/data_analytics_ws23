import math

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
data["Marital_Status"] = data["Marital_Status"].replace({"Alone":"Single","Absurd": "Single", "YOLO": "Single",})

# dropping Data (useless features):

to_drop = ["Z_CostContact", "Z_Revenue", "ID", "Children"]
data = data.drop(to_drop, axis=1)

#  dropping outliers according to original data cleaning, since the variable Year_Birth has been transformed...
# ...differently, we will check if the count is low enough to just drop
print(data["Year_Birth"].value_counts()["The Greatest Generation"])

# three entries, same as in the initial data cleaning- okay to drop

# The count is at 23- not insignificant

data = data[(data["Year_Birth"] != "The Greatest Generation")]
data = data[(data["Income"] < 600000)]
data["Education"] = data["Education"].replace({"2n Cycle": "2nd_Cycle"})


categorical_cols_check = ['Marital_Status', 'Year_Birth', 'Education']

# Distribution of categorical data. Each column plotted
sns.set(style="whitegrid")


# Plotting each categorical column
for col in categorical_cols_check:
    plt.figure(figsize=(10, 5))  # Adjust the size of the figure
    sns.countplot(data=data, x=col, order=data[col].value_counts().index)  # Ordering bars by count
    plt.title(f'Distribution of {col}')
    plt.xticks(rotation=45)  # Rotate x labels for better visibility if needed
    plt.tight_layout()  # Adjust subplot params for better layout
    plt.show()

# "The Silent Generation" count is looks to be quite low

print(data["Year_Birth"].value_counts()["The Silent Generation"])

# The count is at 23- not insignificant-> checking if these are outliers by plotting this data against...
# ...the seemingly most impactful numerical variables: Income and Expenditure. First the variable will be transformed:

# ... changing each entry into a count

year_birth_check = []

for x in data["Year_Birth"]:
    y = data["Year_Birth"].value_counts()[x]
    year_birth_check.append(y)

data["year_birth_check"] = year_birth_check

check = ["Income", "Expenditure", "Household Size"]

for x in check:
    data.plot.scatter("year_birth_check", x)
    plt.show()


# the data seems fine- there seems to be a non linear correlation between the generation and the family size- which does make sense
# this correlation was not checked in the initial data analysis/cleaning
# although it seems like income still has some outliers - the extrem outliers are gone, and these might still be valuable for the data-set.

# checking distributions of numerical data:

numerical_columns_check = ['Income', 'Expenditure', 'Household Size', 'Dt_Customer', 'Kidhome', 'Teenhome', 'Recency',
                  'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds',
                  'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases',
                  'NumWebVisitsMonth']


for col in numerical_columns_check:
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    sns.histplot(data[col], bins=30, kde=True)
    plt.title(f'Distribution of {col}')

    plt.tight_layout()
    plt.show()

    skewness = data[col].skew()
    kurtosis = data[col].kurtosis()
    print(f"{col} - Skewness: {round(skewness, 3)}, Kurtosis: {round(kurtosis, 3)}")

# first address some of the variables with low skewness:
#   -Income is similar to a normal distribution too, with a few outliers-> these will be removed
#   -Kidhome is skewed heavily to the left
#   -Teenhome is skewed heaviliy to the left, too, hence combining these will not work, instead these will be transformed..
#...to a new variable Is Parent- similar to the initial data cleaning
#   -Household size has low skewness and low kurtois- it is similar enough to a normal distribution
#   -Number of Store Purchases is also close enough to a normal distribution
#   -Dt_customer is similar to a univariate distribution, can be transformed with gaussian assumption
#   -Recency is also similar to a univariate distribution, can be transformed with gaussian assumption
#    -Expenditure is skewed to the left heavily, but does not have as high tails
#... NumberWebVisits is close to a normal distribution, however it has quite a few outliers, which will be eliminated
#   -most remaining variables are either heavily skewed, contain a lot of outliers, or both- they do, however, have...
#... the following feature in common: They all have an negative exponential distribution, in order to standardize this...
#... the data is transformed via natural log
#... in general the skewness of the purchasing data (expenditure, products, web visits etc.) suggests that most...
#... customers are don't buy at this store frequently, as the distribution of the number of days of being a customer...
#... are pretty flat, although that would suggest a decline, not an as step as depicted


children = data["Kidhome"]+data["Teenhome"]

is_parent = []

for x in children:
    if x > 0:
        is_parent.append('no')
    else:
        is_parent.append('yes')

data["Is_Parent"] = is_parent

data_to_drop = ["Kidhome", "Teenhome"]
data = data.drop(data_to_drop, axis=1)

# natural log transformation of columns

natural_log_transformations = ['MntWines', 'MntFruits',
                  'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds','NumDealsPurchases',
                  'NumWebPurchases', 'NumCatalogPurchases', 'NumWebVisitsMonth']

def natural_log(x):
    if x == 0:
        y = 0
    else:
        y = math.log(x)
    return y

for x in natural_log_transformations:
    data[x] = data[x].apply(natural_log)


for col in natural_log_transformations:
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    sns.histplot(data[col], bins=30, kde=True)
    plt.title(f'Distribution of {col}')

    plt.tight_layout()
    plt.show()

    skewness = data[col].skew()
    kurtosis = data[col].kurtosis()
    print(f"{col} - Skewness: {round(skewness, 3)}, Kurtosis: {round(kurtosis, 3)}")

# the overall skewness is improved, and the number of outliers is reduced drastically

# due to using a standardized code for the data_transformation, particularly beacuse of some of the binary/categoricak...
# ...variables, the data needs to be transformed from binary to categorical

cat_transform = ['AcceptedCmp3','AcceptedCmp2','AcceptedCmp1','AcceptedCmp4','AcceptedCmp5','Complain', 'Response']

for x in cat_transform:
    data[x] = data[x].replace({1:"yes",0: "no"})

print(data['AcceptedCmp3'])

categorical_cols = ['Marital_Status', 'Year_Birth', 'Is_Parent', 'AcceptedCmp3',
                  'AcceptedCmp2','AcceptedCmp1','AcceptedCmp4','AcceptedCmp5','Complain', 'Response']
ordinal_cols = ['Education']
numerical_cols = ['Income', 'Expenditure', 'Household Size', 'Dt_Customer', 'Recency','MntWines', 'MntFruits',
                  'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds','NumDealsPurchases',
                  'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases','NumWebVisitsMonth']
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
plt.title('KMeans Clustering with 3 Clusters')
plt.show()

## PCA to reduce dimensions and visualise clusters differentiation
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Assuming your data is in a variable called `data`
pca = PCA().fit(data_no_outliers)

# Calculate cumulative sum of explained variances
cum_sum = np.cumsum(pca.explained_variance_ratio_)

# Plot
plt.figure(figsize=(10, 8))
plt.plot(range(1, len(cum_sum) + 1), cum_sum, marker='o', linestyle='--')
plt.title("Cumulative Explained Variance")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.grid(True)
plt.legend(loc="best")
plt.tight_layout()
plt.show()

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

# creating summary of a cluster

data['outliers'] = outliers
data = data[data['outliers']==0]
data['clusters'] = clusters

print(data)

categorical_cols = ['Marital_Status', 'Year_Birth', 'Is_Parent', 'AcceptedCmp3',
                  'AcceptedCmp2','AcceptedCmp1','AcceptedCmp4','AcceptedCmp5','Complain', 'Response']
ordinal_cols = ['Education']
numerical_cols = ['Income', 'Expenditure', 'Household Size', 'Dt_Customer', 'Recency','MntWines', 'MntFruits',
                  'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds','NumDealsPurchases',
                  'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases','NumWebVisitsMonth']

print(data.groupby('clusters').agg({
    'Marital_Status':pd.Series.mode,
    'Year_Birth':pd.Series.mode,
    'Is_Parent':pd.Series.mode,
    'AcceptedCmp3':pd.Series.mode,
    'AcceptedCmp2':pd.Series.mode,
    'AcceptedCmp1':pd.Series.mode,
    'AcceptedCmp4':pd.Series.mode,
    'AcceptedCmp5':pd.Series.mode,
    'Complain':pd.Series.mode,
    'Response':pd.Series.mode,
    'Education':pd.Series.mode,
    'Income':pd.Series.mean,
    'Expenditure':pd.Series.mode,
    'Household Size':pd.Series.median,
    'Recency':pd.Series.median,
    'Dt_Customer':pd.Series.median,
    'MntWines':pd.Series.mean,
    'MntFruits':pd.Series.mean,
    'MntMeatProducts':pd.Series.mean,
    'MntFishProducts':pd.Series.mean,
    'MntSweetProducts':pd.Series.mean,
    'MntGoldProds':pd.Series.mean,
    'NumDealsPurchases':pd.Series.mean,
    'NumWebPurchases':pd.Series.mean,
    'NumCatalogPurchases': pd.Series.mean,
    'NumStorePurchases': pd.Series.mode,
    'NumWebVisitsMonth': pd.Series.mean
    }))