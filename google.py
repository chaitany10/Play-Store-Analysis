import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
warnings.filterwarnings("ignore")
sns.set(style="whitegrid")

from sklearn import metrics
from sklearn.cross_validation import train_test_split
import random
import matplotlib.pyplot as plt

desired_width = 320
pd.set_option('display.width', desired_width)

def change_size(size):
    if 'M' in size:
        x = size[:-1]
        x = float(x)*1000000
        return (x)
    elif 'k' == size[-1:]:
        x = size[:-1]
        x = float(x) *1000
        return (x)
    else:
        return None


df = pd.read_csv("googleplaystore.csv")

df.dropna(inplace=True)

# Cleaning Categories into integers
CategoryString = df["Category"]
categoryVal = df["Category"].unique()
categoryValCount = len(categoryVal)
category_dict = {}
for i in range(0, categoryValCount):
    category_dict[categoryVal[i]] = i
df["Category_c"] = df["Category"].map(category_dict).astype(int)
# df = pd.concat([df,pd.get_dummies(df['Category'],prefix='Category',dummy_na=True)],axis=1).drop(['Category'],axis=1)


df["Size"] = df["Size"].map(change_size)

# filling Size which had NA
df.Size.fillna(method='ffill', inplace=True)

df['Installs'] = [int(i[:-1].replace(',', '')) for i in df['Installs']]


def Evaluationmatrix(y_true, y_predict):
    print('Mean Squared Error: ' + str(metrics.mean_squared_error(y_true, y_predict)))
    print('Mean absolute Error: ' + str(metrics.mean_absolute_error(y_true, y_predict)))
    print('Mean squared Log Error: ' + str(metrics.mean_squared_log_error(y_true, y_predict)))


def Evaluationmatrix_dict(y_true, y_predict, name='Linear - Integer'):
    dict_matrix = {}
    dict_matrix['Series Name'] = name
    dict_matrix['Mean Squared Error'] = metrics.mean_squared_error(y_true, y_predict)
    dict_matrix['Mean Absolute Error'] = metrics.mean_absolute_error(y_true, y_predict)
    dict_matrix['Mean Squared Log Error'] = metrics.mean_squared_log_error(y_true, y_predict)
    return dict_matrix


def type_cat(types):
    if types == 'Free':
        return 0
    else:
        return 1


df['Type'] = df['Type'].map(type_cat)
# Cleaning of content rating classification
RatingL = df['Content Rating'].unique()
RatingDict = {}
for i in range(len(RatingL)):
    RatingDict[RatingL[i]] = i
df['Content Rating'] = df['Content Rating'].map(RatingDict).astype(int)
# dropping of unrelated and unnecessary items
df.drop(labels=['Last Updated', 'Current Ver', 'Android Ver', 'App'], axis=1, inplace=True)
GenresL = df.Genres.unique()
GenresDict = {}
for i in range(len(GenresL)):
    GenresDict[GenresL[i]] = i
df['Genres_c'] = df['Genres'].map(GenresDict).astype(int)


# Cleaning prices
def price_clean(price):
    if price == '0':
        return 0
    else:
        price = price[1:]
        price = float(price)
        return price


df['Price'] = df['Price'].map(price_clean).astype(float)

# convert reviews to numeric
df['Reviews'] = df['Reviews'].astype(int)
df.info()
df2 = pd.get_dummies(df, columns=['Category'])
print(df2.head())

# Integer encoding
X = df.drop(labels=['Category', 'Rating', 'Genres', 'Genres_c'], axis=1)
y = df.Rating
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
model = LinearRegression()
model.fit(X_train, y_train)
Results = model.predict(X_test)

# Creation of results dataframe and addition of first entry
resultsdf = pd.DataFrame()
resultsdf = resultsdf.from_dict(Evaluationmatrix_dict(y_test, Results), orient='index')
resultsdf = resultsdf.transpose()

# dummy encoding

X_d = df2.drop(labels = ['Rating', 'Genres', 'Category_c', 'Genres_c'], axis=1)
y_d = df2.Rating
X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_d, y_d, test_size=0.20)
model_d = LinearRegression()
model_d.fit(X_train_d, y_train_d)
Results_d = model_d.predict(X_test_d)

# adding results into results dataframe
resultsdf = resultsdf.append(Evaluationmatrix_dict(y_test_d, Results_d, name='Linear - Dummy'), ignore_index=True)
print("For Integer Encoding")
Evaluationmatrix(y_test,Results)
print("For dummy encoding")
Evaluationmatrix(y_test_d,Results_d)
plt.figure(figsize=(12, 7))
sns.regplot(Results, y_test, color='teal', label='Integer', marker='x')
sns.regplot(Results_d, y_test_d, color='orange', label='Dummy')
plt.legend()
plt.title('Linear model - Excluding Genres')
plt.xlabel('Predicted Ratings')
plt.ylabel('Actual Ratings')
plt.show()
#Including genre label

#Integer encoding
X = df.drop(labels = ['Category','Rating','Genres'],axis = 1)
y = df.Rating
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
model = LinearRegression()
model.fit(X_train,y_train)
Results = model.predict(X_test)

resultsdf = resultsdf.append(Evaluationmatrix_dict(y_test,Results, name = 'Linear(inc Genre) - Integer'),ignore_index = True)

#dummy encoding

X_d = df2.drop(labels = ['Rating','Genres','Category_c'],axis = 1)
y_d = df2.Rating
X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_d, y_d, test_size=0.30)
model_d = LinearRegression()
model_d.fit(X_train_d,y_train_d)
Results_d = model_d.predict(X_test_d)

resultsdf = resultsdf.append(Evaluationmatrix_dict(y_test_d,Results_d, name = 'Linear(inc Genre) - Dummy'),ignore_index = True)
print("For Integer Encoding")
Evaluationmatrix(y_test,Results)
print("For dummy encoding")
Evaluationmatrix(y_test_d,Results_d)
plt.figure(figsize=(12,7))
sns.regplot(Results,y_test,color='teal', label = 'Integer', marker = 'x')
sns.regplot(Results_d,y_test_d,color='orange',label = 'Dummy')
plt.legend()
plt.title('Linear model - Including Genres')
plt.xlabel('Predicted Ratings')
plt.ylabel('Actual Ratings')
plt.show()

'''For SVM'''

#Excluding genres
from sklearn import svm
#Integer encoding

X = df.drop(labels = ['Category','Rating','Genres','Genres_c'],axis = 1)
y = df.Rating
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

model2 = svm.SVR()
model2.fit(X_train,y_train)

Results2 = model2.predict(X_test)

resultsdf = resultsdf.append(Evaluationmatrix_dict(y_test,Results2, name = 'SVM - Integer'),ignore_index = True)
print("SVR excluding genres integer")
Evaluationmatrix(y_test,Results2)
#dummy based


X_d = df2.drop(labels = ['Rating','Genres','Category_c','Genres_c'],axis = 1)
y_d = df2.Rating

X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_d, y_d, test_size=0.20)

model2 = svm.SVR()
model2.fit(X_train_d,y_train_d)

Results2_d = model2.predict(X_test_d)

resultsdf = resultsdf.append(Evaluationmatrix_dict(y_test_d,Results2_d, name = 'SVM - Dummy'),ignore_index = True)
print("SVR excluding genres dummy")
Evaluationmatrix(y_test_d,Results2_d)
plt.figure(figsize=(12,7))
sns.regplot(Results2,y_test,color='teal', label = 'Integer', marker = 'x')
sns.regplot(Results2_d,y_test_d,color='orange',label = 'Dummy')
plt.legend()
plt.title('SVM model - excluding Genres')
plt.xlabel('Predicted Ratings')
plt.ylabel('Actual Ratings')
plt.show()

#Integer encoding, including Genres_c
model2a = svm.SVR()

X = df.drop(labels = ['Category','Rating','Genres'],axis = 1)
y = df.Rating

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

model2a.fit(X_train,y_train)

Results2a = model2a.predict(X_test)

#evaluation
resultsdf = resultsdf.append(Evaluationmatrix_dict(y_test,Results2a, name = 'SVM(inc Genres) - Integer'),ignore_index = True)
print("SVR including genres integers")
Evaluationmatrix(y_test,Results2a)
#dummy encoding, including Genres_c
model2a = svm.SVR()

X_d = df2.drop(labels = ['Rating','Genres','Category_c'],axis = 1)
y_d = df2.Rating

X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_d, y_d, test_size=0.20)

model2a.fit(X_train_d,y_train_d)

Results2a_d = model2a.predict(X_test_d)

#evaluation
resultsdf = resultsdf.append(Evaluationmatrix_dict(y_test_d,Results2a_d, name = 'SVM(inc Genres) - Dummy'),ignore_index = True)
print("SVR including genres dummy")

Evaluationmatrix(y_test_d,Results2a_d)
plt.figure(figsize=(12,7))
sns.regplot(Results2a,y_test,color='teal', label = 'Integer', marker = 'x')
sns.regplot(Results2a_d,y_test_d,color='orange',label = 'Dummy')
plt.legend()
plt.title('SVM model - including Genres')
plt.xlabel('Predicted Ratings')
plt.ylabel('Actual Ratings')
plt.show()
'''RFR'''

#Integer encoding
X = df.drop(labels = ['Category','Rating','Genres','Genres_c'],axis = 1)
y = df.Rating
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
model3 = RandomForestRegressor()
model3.fit(X_train,y_train)
Results3 = model3.predict(X_test)

#evaluation
resultsdf = resultsdf.append(Evaluationmatrix_dict(y_test,Results3, name = 'RFR - Integer'),ignore_index = True)
print("RFR model including genres integer")
Evaluationmatrix(y_test,Results3)
#dummy encoding

X_d = df2.drop(labels = ['Rating','Genres','Category_c','Genres_c'],axis = 1)
y_d = df2.Rating
X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_d, y_d, test_size=0.20)
model3_d = RandomForestRegressor()
model3_d.fit(X_train_d,y_train_d)
Results3_d = model3_d.predict(X_test_d)

#evaluation
resultsdf = resultsdf.append(Evaluationmatrix_dict(y_test_d,Results3_d, name = 'RFR - Dummy'),ignore_index = True)

plt.figure(figsize=(12,7))
print("RFR model including genres dummy")
Evaluationmatrix(y_test_d,Results3_d)
sns.regplot(Results3,y_test,color='teal', label = 'Integer', marker = 'x')
sns.regplot(Results3_d,y_test_d,color='orange',label = 'Dummy')
plt.legend()
plt.title('RFR model - excluding Genres')
plt.xlabel('Predicted Ratings')
plt.ylabel('Actual Ratings')
plt.show()

#Including Genres_C

#Integer encoding
X = df.drop(labels = ['Category','Rating','Genres'],axis = 1)
y = df.Rating
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
model3a = RandomForestRegressor()
model3a.fit(X_train,y_train)
Results3a = model3a.predict(X_test)

#evaluation
resultsdf = resultsdf.append(Evaluationmatrix_dict(y_test,Results3a, name = 'RFR(inc Genres) - Integer'),ignore_index = True)
Evaluationmatrix(y_test,Results3a)
#dummy encoding

X_d = df2.drop(labels = ['Rating','Genres','Category_c'],axis = 1)
y_d = df2.Rating
X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_d, y_d, test_size=0.20)
model3a_d = RandomForestRegressor()
model3a_d.fit(X_train_d,y_train_d)
Results3a_d = model3a_d.predict(X_test_d)

#evaluation
resultsdf = resultsdf.append(Evaluationmatrix_dict(y_test_d,Results3a_d, name = 'RFR(inc Genres) - Dummy'),ignore_index = True)
Evaluationmatrix(y_test_d,Results3a_d)
resultsdf.set_index('Series Name', inplace = True)

plt.figure(figsize = (10,12))
plt.subplot(3,1,1)
resultsdf['Mean Squared Error'].sort_values(ascending = False).plot(kind = 'barh',color=(0.3, 0.4, 0.6, 1), title = 'Mean Squared Error')
plt.subplot(3,1,2)
resultsdf['Mean Absolute Error'].sort_values(ascending = False).plot(kind = 'barh',color=(0.5, 0.4, 0.6, 1), title = 'Mean Absolute Error')
plt.subplot(3,1,3)
resultsdf['Mean Squared Log Error'].sort_values(ascending = False).plot(kind = 'barh',color=(0.7, 0.4, 0.6, 1), title = 'Mean Squared Log Error')
plt.show()
plt.savefig('final.png')