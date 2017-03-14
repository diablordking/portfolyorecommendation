from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_extraction import text

data = pd.read_csv('input/Combined_News_DJIA.csv')
#print(data)

train = data[data['Date'] < '2015-01-01']
test = data[data['Date'] > '2014-12-31']
poly = PolynomialFeatures(degree=2)


trainheadlines = []
for row in range(0,len(train.index)):
    trainheadlines.append(' '.join(str(x) for x in train.iloc[row,2:27]))
    #print(train.iloc[row,2:27])
advancedvectorizer = CountVectorizer(ngram_range=(2,2))
#advancedtrain = poly.fit_transform(trainheadlines)
advancedtrain = advancedvectorizer.fit_transform(trainheadlines)

print(advancedtrain)

advancedmodel = LogisticRegression()
advancedmodel = advancedmodel.fit(advancedtrain, train["Label"])
print(advancedmodel)
testheadlines = []
for row in range(0,len(test.index)):
    testheadlines.append(' '.join(str(x) for x in test.iloc[row,2:27]))
advancedtest = advancedvectorizer.transform(testheadlines)
     #advancedtest=advancedtest.stop_words("english")
advpredictions = advancedmodel.predict(advancedtest)

print(pd.crosstab(test["Label"], advpredictions, rownames=["Actual"], colnames=["Predicted"]))



advwords = advancedvectorizer.get_feature_names()
advcoeffs = advancedmodel.coef_.tolist()[0]
advcoeffdf = pd.DataFrame({'Words' : advwords,
                        'Coefficient' : advcoeffs})
advcoeffdf = advcoeffdf.sort_values(['Coefficient', 'Words'], ascending=[0, 1])

print(advcoeffdf.head(10))


