#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns

get_ipython().run_line_magic('run', 'clean_data.ipynb')

data = pd.read_csv('./data/medicalCassificationPart2-add.csv',sep=";", encoding='cp1250')
data_copi = data.copy()

clean_data = clean_dataFrame(data)
df = dataFrame_matrix(clean_data)
df2 =dropna_columns(get_name(dic_col),clean_data)
finish_dataFrame =connect_df(df,df2)


# In[705]:


finish_dataFrame.shape 


# In[ ]:





# In[652]:


corr =finish_dataFrame.corr('pearson')
for x in corr['class'].items():
    if pd.isnull(x[1]):
        print(x)


# In[10]:


cor_target = abs (corr['class'])
relevant_features = cor_target [cor_target> 0.] 

print(len(relevant_features))
v =sorted(relevant_features.items(), key=lambda x: (-x[1], x[0]))
for x,y in enumerate(v):
    print(y[0])


# In[11]:



cor =finish_dataFrame[['class' ,'awareness_8','HR','SPO2 ','RR','awareness_0','EMS','place _8','awareness_1','breath_0','main symptoms_0','skin_9','additional information_0','main symptoms_37','main symptoms_39']].corr('spearman')
plt.figure(figsize=(12,10)),
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()


# In[5]:


# rozdzielenie cech i etykiet
X= finish_dataFrame.drop(['class'], axis=1)
fetures = X.columns
y=finish_dataFrame['class'].values


# In[ ]:





# In[8]:


from sklearn.linear_model import LogisticRegression

#wyuczenie modelu RandomForest
#model_RF = RandomForestClassifier(n_estimators=20, max_depth=10)
model_RF = RandomForestClassifier(max_features='sqrt', min_samples_split=5, max_depth= 5,
                      n_estimators=600, random_state=8)
#model_RF = RandomForestClassifier(n_estimators= 400, min_samples_split=5, min_samples_leaf= 1, max_features= 'sqrt', max_depth= 40, bootstrap= False)

model_tree = tree.DecisionTreeClassifier(min_samples_split=5, random_state=99, max_depth=100)
#model_Reg=LogisticRegression()
model_Reg=LogisticRegression(C=0.9, class_weight='balanced', multi_class='multinomial',
                  random_state=8, solver='saga')

   


# In[12]:


list_import_name=[]
list_import_val=[]
# define dataset
#X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, random_state=1)
# define the model
model = model_RF
# fit the model
model.fit(X, y)
# get importance
importance = model.feature_importances_
indices = np.argsort(importance)[::-1]

for x in range(X.shape[1]):
    if importance[indices[x]]>0.006:
        list_import_name.append(fetures[indices[x]])
        list_import_val.append(importance[indices[x]])
        
plt.figure(figsize=(14,7))
plt.bar(list_import_name,list_import_val)
plt.xticks(rotation='vertical',fontsize=10)
plt.show()
print(len(list_import_val))


# In[15]:


fit_tree = model_tree.fit(X, y)
list_import_name=[]
list_import_val=[]
y_val=[]
importances = fit_tree.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(X.shape[1]):
    if importances[indices[f]]:
        list_import_name.append(fetures[indices[f]])
        list_import_val.append(importances[indices[f]])
        #print("%2d) %-*s %f" % (f + 1, 30, fetures[indices[f]], importances[indices[f]]))  
print(list_import_name)
print(len(list_import_name))


# In[ ]:





# In[659]:


import numpy as np
import matplotlib.pyplot as plt


# In[660]:


cut_feature_val = pd.DataFrame(X[list_import_name[:45]],columns=list_import_name[:120])


# In[669]:





# In[662]:




# linear regression feature importance
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot
list_import_name=[]
list_import_val=[]
# define dataset
#X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, random_state=1)
# define the model
model = LinearRegression()
# fit the model
model.fit(X, y)
# get importance
importance = model.coef_
war_bez_importnce = np.abs(importance)
#war_bez_importnce = importance

indices = np.argsort(war_bez_importnce)[::-1]
for x in range(X.shape[1]):
    if war_bez_importnce[indices[x]]>2:
        list_import_name.append(fetures[indices[x]])
        list_import_val.append(war_bez_importnce[indices[x]])
plt.figure(figsize=(14,7))
plt.bar(list_import_name,list_import_val)
plt.xticks(rotation='vertical',fontsize=10)
#plt.xticks([])
plt.show()
print(len(list_import_val))


# In[663]:


#selectModel
finish_dataFrame
fetures = X.columns

embeded_rf_selector = SelectFromModel(model_Reg, max_features=50)
embeded_rf_selector.fit(X.loc[:,fetures].values, y)
print( embeded_rf_selector.threshold_)
print(embeded_rf_selector)
importance =embeded_rf_selector.estimator_.coef_[0]
list_import_name=[]
list_import_val=[]
war_bez_importnce = np.abs(importance)
#war_bez_importnce = importance

indices = np.argsort(war_bez_importnce)[::-1]

for x in range(X.shape[1]):
    if war_bez_importnce[indices[x]]:
        list_import_name.append(fetures[indices[x]])
        list_import_val.append(war_bez_importnce[indices[x]])
plt.figure(figsize=(14,7))
plt.bar(list_import_name,list_import_val)
plt.xticks(rotation='vertical',fontsize=10)
#plt.xticks([])
plt.show()
print(len(list_import_val))

#print(embeded_rf_selector.estimator_.coef_[4])
    #index_importances = np.argsort(embeded_rf_selector.estimator_.feature_importances_)[::-1]
    #importantes = embeded_rf_selector.estimator_.feature_importances_
    #for x in  range(X.shape[1]):
     #    print("%2d) %-*s %f" % (x + 1, 30, fetures[index_importances[x]], importantes[index_importances[x]])) 
   # print(embeded_rf_selector.estimator_.feature_importances_)
   
embeded_rf_support = embeded_rf_selector.get_support()
embeded_rf_feature = X.loc[:,embeded_rf_support].columns.tolist()

print(str(len(embeded_rf_feature)),'selected features')
print(embeded_rf_feature)


# In[4]:


#select model
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
finish_dataFrame
fetures = X.columns
list_import_name=[]
list_import_val=[]
for model in [model_RF,model_tree]:
    embeded_rf_selector = SelectFromModel(model, max_features=50)
    embeded_rf_selector.fit(X.loc[:,fetures].values, y)
    print( embeded_rf_selector.threshold_)
    #print(embeded_rf_selector.estimator_.feature_importances_)

    index_importances = np.argsort(embeded_rf_selector.estimator_.feature_importances_)[::-1]
    importantes = embeded_rf_selector.estimator_.feature_importances_
    #for x in  range(X.shape[1]):
     #    print("%2d) %-*s %f" % (x + 1, 30, fetures[index_importances[x]], importantes[index_importances[x]])) 
   # print(embeded_rf_selector.estimator_.feature_importances_)
   
    embeded_rf_support = embeded_rf_selector.get_support()
    embeded_rf_feature = X.loc[:,embeded_rf_support].columns.tolist()
    print(str(len(embeded_rf_feature)),'selected features')
    print(embeded_rf_feature)
plt.plot(list_import_name,list_import_val)
plt.xticks(rotation='vertical',fontsize=7.5)
plt.show()


# In[ ]:





# In[628]:


from sklearn.ensemble import RandomForestClassifier
#model = RandomForestClassifier(n_estimators=100, max_depth=5)

# Train
model_RF.fit(cut_feature_val, y)
# Extract single tree
estimator = model_RF.estimators_[1]
estimator
dot_data = tree.export_graphviz(estimator, out_file=None, feature_names=cut_feature_val.columns, 
                                class_names=['1', '2','3','4','5'],
                                rounded=True, filled=True,
                               special_characters=True) #Gini decides which attribute/feature should be placed at the root node, which features will act as internal nodes or leaf nodes
#Create Graph from DOT data
graph = pydotplus.graph_from_dot_data(dot_data)

# Show graph
Image(graph.create_png())


# In[7]:



from sklearn import tree #For our Decision Tree
import pydotplus # To create our Decision Tree Graph
from IPython.display import Image  
clf = model_tree
#y = drop_class
#X = drop_1
clf_train = clf.fit(X, y)

#print(tree.export_graphviz(clf_train, None))

#Create Dot Data
dot_data = tree.export_graphviz(clf_train, out_file=None, feature_names=fetures, 
                                class_names=['1', '2','3','4','5'], rounded=True, filled=True) #Gini decides which attribute/feature should be placed at the root node, which features will act as internal nodes or leaf nodes
#Create Graph from DOT data
graph = pydotplus.graph_from_dot_data(dot_data)

# Show graph
Image(graph.create_png())


# In[ ]:





# In[397]:


embeded_rf_selector = SelectFromModel(tree.DecisionTreeClassifier(min_samples_split=20, random_state=99), max_features=len(finish_dataFrame))
embeded_rf_selector.fit(X.loc[:,fetures].values, y)
embeded_rf_support = embeded_rf_selector.get_support()
embeded_rf_feature = X.loc[:,embeded_rf_support].columns.tolist()
print(str(len(embeded_rf_feature)),'selected features')
print(embeded_rf_feature)


# In[716]:


cut_feature_val = pd.DataFrame(X[list_import_name[:45]],columns=list_import_name[:120])
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
proj = pca.fit_transform(X)
plt.scatter(proj[:, 0], proj[:, 1], c=y) 

plt.colorbar() 


# In[ ]:





# In[ ]:





# In[ ]:




