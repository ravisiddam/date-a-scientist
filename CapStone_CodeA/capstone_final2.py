
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC


# In[2]:


#Load dataset
df = pd.read_csv('profiles.csv')


# In[ ]:


df.columns


# In[ ]:


essay_cols = ["essay0", "essay1", "essay2", "essay3", "essay4", "essay5", "essay6", "essay7", "essay8", "essay9"]
zodsign = list(['aries', 'taurus', 'gemini', 'cancer', 'leo', 'virgo', 'libra', 'scorpio', 'sagittarius', 'capricorn', 'aquarius', 'pisces'])
diet_cats = { "anything":"nonveg", "halal":"nonveg", "kosher":"nonveg", "mostly anything":"nonveg", "mostly vegetarian":"veg", "mostly other":"nonveg", "mostly vegan":"veg", "mostly kosher":"nonveg", "mostly halal":"nonveg", "strictly anything":"nonveg",
 "strinctly other":"nonveg", "strictly kosher":"nonveg", "strictly halal":"nonveg", "strictly vegan":"veg", "strictly vegeterian":"veg", "vegan":"veg", "vegeterian":"veg", "other":"nonveg", "NaN":"nonveg"}


# In[ ]:


drink_mapping = {"not at all": 0, "rarely": 1, "socially": 2, "often": 3, "very often": 4, "desperately": 5}
smokes_mapping = {"no": 0, "sometimes": 1, "when drinking": 2, "yes": 3, "trying to quit": 4}
drugs_mapping = {"never": 0, "sometimes": 1, "often": 2}
diet_mapping = {"veg":0, "nonveg":1}
body_cat_mapping = {"ectomorph":0, "endomorph":1, "average":2, "mesomorph":3, "others":4}
essay_cols = ["essay0", "essay1", "essay2", "essay3", "essay4", "essay5", "essay6", "essay7", "essay8", "essay9"]
sign_cat_mapping = {'aries':0, 'taurus':1, 'gemini':2, 'cancer':3, 'leo':4, 'virgo':5, 'libra':6, 'scorpio':7, 'sagittarius':8, 'capricorn':9, 'aquarius':10, 'pisces':11}


# In[ ]:


new_sign = []
for zod in zodsign:
    for i in range(100):
        if zod in str(df.sign[i]):
           # print(df.sign[i])else:
            new_sign.append(zod)


# In[ ]:


new_signs = df.sign
new_signs = new_signs.replace(' and it&rsquo;s fun to think about', '', regex=True)
new_signs = new_signs.replace(' but it doesn&rsquo;t matter', '', regex=True)
new_signs = new_signs.replace(' and it matters a lot', '', regex=True)
new_signs = new_signs.replace('NaN', '', regex=True)
df['sign_cat'] = new_signs


# In[ ]:


df["diet_cat"] = df["diet"].map(diet_cats)


# In[ ]:


df["diet_cat"] = df["diet"].map(diet_cats)


# In[ ]:


df.columns


# In[ ]:


#all_data = df.iloc[:, [0,3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 28, 31, 32, 33, 34, 35]]
all_data = df.iloc[:, [3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 28, 31, 32]]
all_data.columns


# In[ ]:


# Removing the NaNs
all_essays = all_data[essay_cols].replace(np.nan, '', regex=True)
# Combining the essays
all_essays = all_essays[essay_cols].apply(lambda x: ' '.join(x), axis=1)


# In[ ]:


all_essays[34513] = ''
all_essays[40640] = ''
all_essays[44214] = ''
all_essays[52093] = ''
all_essays[58762] = ''
all_essays[58866] = ''
all_essays[59945] = ''


# In[ ]:


all_data['drinks_code'] = all_data.drinks.map(drink_mapping)
all_data['smokes_code'] = all_data.smokes.map(smokes_mapping)
all_data['drugs_code'] = all_data.drugs.map(drugs_mapping)
all_data['sign_code'] = all_data.sign_cat.map(sign_cat_mapping)
all_data['diet_code'] = all_data.diet_cat.map(diet_mapping)
all_data["essay_len"] = all_essays.apply(lambda x: len(x))
all_data.head(10)


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
cv =  CountVectorizer(stop_words=None, analyzer='word')

def matrix_to_list(matrix):
    matrix = matrix.toarray()
    return matrix.tolist()

def avg_mei_count(essays):
    temp_essays = essays
    avg_wordlist = []
    temp1 = [temp_essays]
    for t in range(1):
        cv_score = cv.fit_transform(temp1)
        cv_score_list = matrix_to_list(cv_score)
        cv_wordlist = cv.get_feature_names()
        for i in range(len(cv_wordlist)):
            wordlen = [len(word) for word in cv_wordlist]
  
            #print(cv_score_list[0][cv.get_feature_names().index('i')] + cv_score_list[0][cv.get_feature_names().index('me')])
        meic = 0
        if 'me' in cv.get_feature_names():
            meic += cv_score_list[0][cv.get_feature_names().index('me')]
        if 'i'  in cv.get_feature_names():
            meic += cv_score_list[0][cv.get_feature_names().index('i')]
        return (meic, np.mean(wordlen)) 


# In[ ]:


get_ipython().run_line_magic('timeit', '')
avglen = []
meicount = []
wordcount = []
for i in range(len(all_essays)):
    wordcount.append(len(all_essays[i].split()))
    if len(all_essays[i].split())>2:
        a, b = avg_mei_count(all_essays[i])
    else:
        a = 0
        b = 0
    meicount.append(a)
    avglen.append(b)


# In[ ]:


all_data["avg_word_length"] = avglen
all_data["me_i_count"] = meicount


# In[ ]:


all_data['essay_word_count'] = wordcount


# In[ ]:


#df["diet_cat"] = df["diet"].map(diet_cats)


# In[ ]:


all_data.columns


# In[ ]:


all_data['drugs_code'] = all_data['drugs_code'].fillna(all_data.drugs_code.mean())
all_data['smokes_code'] = all_data['smokes_code'].fillna(all_data.drugs_code.mean())
all_data['drinks_code'] = all_data['drinks_code'].fillna(all_data.drugs_code.mean())
all_data['diet_code'] = all_data['diet_code'].fillna(1)
#all_data['body_code'] = all_data['body_code'].fillna(4)
all_data['sign_code'] = all_data['sign_code'].fillna(all_data.drugs_code.mean())


# In[ ]:


all_data.to_csv('final_profiles.csv', index=False)


# In[3]:


all_data = pd.read_csv('final_profiles.csv')
all_data.shape


# In[4]:


all_data.columns


# In[5]:


feature_data = all_data[['smokes_code', 'drinks_code', 'drugs_code', 'essay_len', 'avg_word_length']]


# In[6]:


feature_data.shape


# In[7]:


x = feature_data.values
min_max_scaler = MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)


# In[ ]:


#x_scaled[0:10]


# In[8]:


feature_data = pd.DataFrame(x_scaled, columns=feature_data.columns)


# In[9]:


feature_data.head(20)


# In[10]:


X = feature_data.values
y = all_data['sign_code']
z = all_data['diet_code']


# In[11]:


X.shape


# In[12]:


y = y.astype(int)
type(y)


# In[ ]:


y.describe()


# In[ ]:


z.describe()


# In[13]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, test_size = 0.2, random_state=6)
x_train, x_test, z_train, z_test = train_test_split(X, z, train_size = 0.8, test_size = 0.2, random_state=6)


# In[14]:


# Spot-Check Algorithms for classify zodiac sign using smokes, drinks, drugs, essay length, avg word length
models = []
models.append(('LR', LogisticRegression()))
#models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
#models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=10, random_state=6)
    cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


# In[15]:


# Spot-Check Algorithms classify diet code using smokes, drinks, drugs, essay length, avg word length
models = []
models.append(('LR', LogisticRegression()))
#models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=10, random_state=6)
    cv_results = cross_val_score(model, x_train, z_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


# In[ ]:


all_data.columns


# In[16]:


import matplotlib.pyplot as plt
scores = []
for k in range(1, 100):
    classifier = KNeighborsClassifier(n_neighbors = k)
    classifier.fit(x_train, z_train)
    scores.append(classifier.score(x_test, z_test))
    
plt.plot(range(1, 100), scores)
plt.show()


# In[ ]:


#zod_df_x = df[['body_code', 'diet_code', 'drugs_code', 'smokes_code', 'drinks_code']]


# In[ ]:


#all_data['diet_code'] = df.diet_cat.map(diet_mapping)
#all_data.columns


# In[ ]:


X_dat = all_data[['smokes_code', 'drinks_code', 'drugs_code', 'diet_code']]
Y_dat = all_data['sign_code']
X_dat.shape


# In[ ]:


smokes_dummies    = pd.get_dummies(all_data['smokes_code'], prefix='smokes', drop_first=True)
drinks_dummies    = pd.get_dummies(all_data['drinks_code'], prefix='drinks', drop_first=True)
drugs_dummies     = pd.get_dummies(all_data['drugs_code'], prefix='drugs', drop_first=True)
diet_dummies     = pd.get_dummies(all_data['diet_code'], prefix='diet', drop_first=True)
x_new = pd.concat([smokes_dummies, drinks_dummies, drugs_dummies, diet_dummies], axis=1)

#sign_dummies      = pd.get_dummies(all_data['sign_code'], prefix='sign', drop_first=True)
    


# In[ ]:


x_new.head()


# In[ ]:


#feature_data.columns
x_new['essay_len'] = feature_data['essay_len']
#x_new['avg_word_length'] = feature_data['avg_word_length']


# In[ ]:


#from sklearn.preprocessing import OneHotEncoder
#enc = OneHotEncoder()
#X_bin = enc.fit_transform(X_dat).toarray()


# In[ ]:


#X_bin[0:10]


# In[ ]:


xnewx = x_new.values
y = y.astype(int)


# In[ ]:


from sklearn.model_selection import train_test_split
x_tr, x_ts, y_tr, y_ts = train_test_split(xnewx, y, train_size = 0.8, test_size = 0.2, random_state=6)


# In[ ]:


import matplotlib.pyplot as plt
scores = []
for k in range(1, 100):
    classifier = KNeighborsClassifier(n_neighbors = k)
    classifier.fit(x_tr, y_tr)
    scores.append(classifier.score(x_ts, y_ts))
    
plt.plot(range(1, 100), scores)
plt.show()


# In[ ]:


plt.plot(range(50,100), scores)
plt.show()


# In[ ]:


nbc = MultinomialNB()
nbc.fit(x_tr, y_tr)
pred = nbc.predict(x_ts)
accuracy_score(y_ts, pred)


# In[ ]:


df['me_i_count'] = all_data['me_i_count'] 
#yreg = df['age']
regdata = df[['me_i_count', 'age']]
regdata.shape


# In[ ]:


regdata2 = regdata[regdata['age'] < 45]
regdata2 = regdata2[regdata2['age'] > 18]
#regdata2 = regdata2[regdata2['me_i_count'] > 20]
regdata2.shape


# In[ ]:


#Linear Regression model to predict Age using Me or I count in the essay
from sklearn.linear_model import LinearRegression
xreg = np.array(regdata2[['me_i_count']])
yreg = np.array(regdata2[['age']])
#xreg_tr, xreg_ts, yreg_tr, yreg_ts = train_test_split(xreg, yreg, test_size=0.2, random_state=1)
xreg = xreg.reshape(-1, 1)
#xreg = xreg.reshape(-1, 1)
lr = LinearRegression()
lr.fit(xreg, yreg)
lrpred = lr.predict(xreg)
print(lr.coef_)
print(lr.intercept_)
plt.plot(xreg, yreg, 'o')
plt.xlabel('Me or I count')
plt.ylabel('age')
plt.plot(xreg, lrpred)
plt.show()


# In[ ]:



yreg = yreg.ravel()
from sklearn.svm import SVR
svr_poly = SVR(kernel='poly', C=1e3, degree=2)
y_poly = svr_poly.fit(xreg, yreg).predict(xreg)
plt.plot(xreg, y_poly, color='cornflowerblue', lw=lw, label='Polynomial model')
plt.xlabel('data')
plt.ylabel('target')
plt.title('Support Vector Regression')
plt.legend()
plt.show()


# In[ ]:


#Regression Result: 
#linear regression is not impressive 
#Support Vector Regression is 

