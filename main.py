import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#matplotlib inline
import seaborn as sns
from sklearn import metrics 
import warnings
warnings.filterwarnings('ignore')



data = pd.read_csv("phishing.csv")
data.head()

data.shape
data.columns
data.info()
data.nunique()
data = data.drop(['Index'],axis = 1)
data.describe().T



#plt.figure(figsize=(15,15))
#sns.heatmap(data.corr(), annot=True)
#plt.show()


#df = data[['PrefixSuffix-', 'SubDomains', 'HTTPS','AnchorURL','WebsiteTraffic','class']]
#sns.pairplot(data = df,hue="class",corner=True);


#data['class'].value_counts().plot(kind='pie',autopct='%1.2f%%')
#plt.title("Phishing Count")
#plt.show()


X = data.drop(["class"],axis =1)
y = data["class"]


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
X_train.shape, y_train.shape, X_test.shape, y_test.shape



# Creating holders to store the model performance results
ML_Model = []
accuracy = []
f1_score = []
recall = []
precision = []

#function to call for storing the results
def storeResults(model, a,b,c,d):
  ML_Model.append(model)
  accuracy.append(round(a, 3))
  f1_score.append(round(b, 3))
  recall.append(round(c, 3))
  precision.append(round(d, 3))


  # Linear regression model 
from sklearn.linear_model import LogisticRegression
#from sklearn.pipeline import Pipeline

# instantiate the model
log = LogisticRegression()

# fit the model 
log.fit(X_train,y_train)




y_train_log = log.predict(X_train)
y_test_log = log.predict(X_test)
