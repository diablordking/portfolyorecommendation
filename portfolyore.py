import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

data = pd.read_csv('input/Combined_News_DJIA.csv')

data.head()
print(data.head())

train = data[data['Date'] < '2015-01-01']
test = data[data['Date'] > '2014-12-31']



example = train.iloc[3,10]
print(example)

example2 = example.lower()
print(example2)

example3 = CountVectorizer().build_tokenizer()(example2)
print(example3)

print(pd.DataFrame([[x,example3.count(x)] for x in set(example3)], columns = ['Word', 'Count']))


trainheadlines = []
for row in range(0,len(train.index)):
    trainheadlines.append(' '.join(str(x) for x in train.iloc[row,2:27]))

print(trainheadlines)

basicvectorizer = CountVectorizer()
basictrain = basicvectorizer.fit_transform(trainheadlines)
print(basictrain.shape)

basicmodel = LogisticRegression()
basicmodel = basicmodel.fit(basictrain, train["Label"])

testheadlines = []
for row in range(0,len(test.index)):
    testheadlines.append(' '.join(str(x) for x in test.iloc[row,2:27]))
basictest = basicvectorizer.transform(testheadlines)
predictions = basicmodel.predict(basictest)


print(pd.crosstab(test["Label"], predictions, rownames=["Actual"], colnames=["Predicted"]))


basicwords = basicvectorizer.get_feature_names()
basiccoeffs = basicmodel.coef_.tolist()[0]
coeffdf = pd.DataFrame({'Word' : basicwords,
                        'Coefficient' : basiccoeffs})
coeffdf = coeffdf.sort_values(['Coefficient', 'Word'], ascending=[0, 1])


print(coeffdf.head(10))
