import math
import pandas as pd

# reading data from csv into pandas dataframe
data = pd.read_csv("./data/marketing_campaign.csv", sep=";")
data = data.dropna()  # drop null values, there is only 24 of them

# changing type to datetime
data["Dt_Customer"] = pd.to_datetime(data["Dt_Customer"], dayfirst=True)

# creating a list to store dates as a placeholder, for transformation of the data
customer_join_dates = []  
for x in data["Dt_Customer"]:
    x = x.date()
    customer_join_dates.append(x)

reference_customer = max(customer_join_dates)

days = []  # transformed Dt_Customer variable
for y in customer_join_dates:
    number_of_days = reference_customer-y
    days.append(number_of_days)

data["Dt_Customer"] = days  # overwrites column
data["Dt_Customer"] = data["Dt_Customer"].dt.days  # changes column to the birth year was changed into age in the data cleaning

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

# checking in which generation each person falls
generations_list = []
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

# transforming the data
data["Year_Birth"] = generations_list
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


# The count is at 23- not insignificant
data = data[(data["Year_Birth"] != "The Greatest Generation")]
data = data[(data["Income"] < 600000)]
data["Education"] = data["Education"].replace({"2n Cycle": "2nd_Cycle"})


categorical_cols_check = ['Marital_Status', 'Year_Birth', 'Education']
# The count is at 23- not insignificant-> checking if these are outliers by plotting this data against...
# ...the seemingly most impactful numerical variables: Income and Expenditure. First the variable will be transformed:

# ... changing each entry into a count
year_birth_check = []
for x in data["Year_Birth"]:
    y = data["Year_Birth"].value_counts()[x]
    year_birth_check.append(y)

data["year_birth_check"] = year_birth_check

check = ["Income", "Expenditure", "Household Size"]
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

cat_transform = ['AcceptedCmp3','AcceptedCmp2','AcceptedCmp1','AcceptedCmp4','AcceptedCmp5','Complain', 'Response']

for x in cat_transform:
    data[x] = data[x].replace({1:"yes",0: "no"})

data.to_csv('./data/customer_segmentation_clean.csv', sep= ';')