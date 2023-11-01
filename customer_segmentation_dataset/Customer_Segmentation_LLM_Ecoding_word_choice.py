import math

import pandas as pd # for dataframe manipulation

# reading data from csv into pandas dataframe
data = pd.read_csv("marketing_campaign.csv", sep=";")


data = data.dropna()  # since there are only 24 null entries, it is best to drop this data, rather than estimating...
# ...some filler values through a categorical average or regression analysis


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

data["Generation"] = data["Year_Birth"]

to_drop = ["Z_CostContact", "Z_Revenue", "ID", "Children", "Year_Birth"]
data = data.drop(to_drop, axis=1)


data = data[(data["Generation"] != "The Greatest Generation")]
data = data[(data["Income"] < 600000)]
data["Education"] = data["Education"].replace({"2n Cycle": "2nd_Cycle"})


categorical_cols_check = ['Marital_Status', 'Generation', 'Education']

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


data['Accepted 3rd Campaign'] = data['AcceptedCmp3']
data['Accepted 4th Campaign'] = data['AcceptedCmp4']
data['Accepted 5th Campaign'] = data['AcceptedCmp5']
data['Accepted 2nd Campaign'] = data['AcceptedCmp2']
data['Accepted 1st Campaign'] = data['AcceptedCmp1']
data['Accepted last Campaign'] = data['Response']

data_to_drop_cat = ['AcceptedCmp3','AcceptedCmp2','AcceptedCmp1','AcceptedCmp4','AcceptedCmp5', 'Response']
data = data.drop(data_to_drop_cat, axis=1)

cat_transform = ['Accepted 3rd Campaign','Accepted 4th Campaign','Accepted 5th Campaign','Accepted 2nd Campaign','Accepted 1st Campaign','Complain', 'Accepted last Campaign']

for x in cat_transform:
    data[x] = data[x].replace({1:f"{x}",0: f"not"})


def transform_to_embeddings(df: pd.DataFrame, model: object) -> pd.DataFrame:
    """This function takes a dataframe and creates a new dataframe with emeddings

    Args:
        df (pd.DataFrame): The dataframe with input data
        model (object): A model from huggingface to encode data

    Returns:
        df_embedding (pd.DataFrame): df with encoded data
    """
    sentences = []
    for _, row in df.iloc[:, 1:].iterrows():
        sent = []
        for col in df.columns[1:]:
            row_as_sent = f"{col} : {row[col]}"
            sent.append(row_as_sent)
        sentences.append(' '.join(sent))

    output = model.encode(sentences=sentences, show_progress_bar=True, normalize_embeddings=True)
    df_embedding = pd.DataFrame(output)

    return df_embedding

from sentence_transformers import SentenceTransformer
model = SentenceTransformer(r'sentence-transformers/paraphrase-MiniLM-L6-v2')

embedded_data = transform_to_embeddings(data, model)
embedded_data

from pyod.models.ecod import ECOD
clf = ECOD()
outliers = clf.fit_predict(embedded_data)

embedded_data['outliers'] = outliers
embedded_data_no_outliers = embedded_data[embedded_data['outliers']==0].drop(columns=['outliers'])
embedded_data_no_outliers.shape

from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer

km = KMeans(init="k-means++", random_state=0, n_init="auto")
visualizer = KElbowVisualizer(km, k=(2, 10))

visualizer.fit(embedded_data_no_outliers)
visualizer.show()

km = KMeans(n_clusters=5, init='k-means++', random_state=42, n_init = 10, max_iter=10000)
clusters = km.fit_predict(embedded_data_no_outliers)

print(km.cluster_centers_)

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Reduce data to 2D
pca = PCA(n_components=2)
data_pca = pca.fit_transform(embedded_data_no_outliers)

# Plot
plt.scatter(data_pca[:, 0], data_pca[:, 1], c=clusters, cmap='rainbow', edgecolor='k', s=100)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('KMeans Clustering with 5 Clusters')
plt.show()

# Reduce data to 3D
pca = PCA(n_components=3)
data_pca = pca.fit_transform(embedded_data_no_outliers)

# 3D-Plot
graph = plt.figure()
axis = graph.add_subplot(projection="3d")
xdata = data_pca[:, 0]
ydata = data_pca[:, 1]
zdata = data_pca[:, 2]
axis.scatter3D(xdata, ydata, zdata, c=clusters, cmap = "rainbow", s=80)
axis.set_xlabel("Principal Component 1")
axis.set_ylabel("Principal Component 2")
axis.set_zlabel("Principal Component 3")
plt.show()

data['outliers'] = outliers
data = data[data['outliers']==0]
data['clusters'] = clusters

print(data.groupby('clusters').agg({
    'Marital_Status':pd.Series.mode,
    'Generation':pd.Series.mode,
    'Is_Parent':pd.Series.mode,
    'Accepted 3rd Campaign':pd.Series.mode,
    'Accepted 4th Campaign':pd.Series.mode,
    'Accepted 5th Campaign':pd.Series.mode,
    'Accepted 2nd Campaign':pd.Series.mode,
    'Accepted 1st Campaign':pd.Series.mode,
    'Complain':pd.Series.mode,
    'Accepted last Campaign':pd.Series.mode,
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