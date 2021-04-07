#!/usr/bin/env python
# coding: utf-8

# In[246]:


get_ipython().run_line_magic('run', 'Class_Clean_data.ipynb')

data = pd.read_csv('./data/medicalCassificationPart2-add.csv',sep=";", encoding='cp1250')
data_copi = data.copy()
clean_data = clean_dataFrame(data)
df = dataFrame_matrix(clean_data)
df2 =dropna_columns(get_name(dic_col),clean_data)
finish_dataFrame =connect_df(df,df2)


# In[ ]:





# In[14]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import average_precision_score


import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from pprint import pprint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import ShuffleSplit

from sklearn.feature_selection import SelectFromModel


import seaborn as sns
import time
import datetime as dt


# In[ ]:





# In[235]:


#finish_dataFrame =finish_dataFrame.drop([146])
#finish_dataFrame = finish_dataFrame.reset_index()


# In[ ]:





# In[247]:


finish_dataFrame = finish_dataFrame.drop(['main symptoms_13','main symptoms_18','main symptoms_26','medical history_5','medical history_30','awareness_7', 'awareness_18','skin_21'], axis:=1)


# In[248]:


y =finish_dataFrame['class'].values
features_243 = finish_dataFrame.drop(['class'], axis:=1).columns
features_6_pow4=['HR','RR','SPO2 ','awareness_0','awareness_8','EMS']
features_6_pow4_new_feature=['HR','RR','SPO2 ','awareness_0','awareness_8','EMS','skin_0','place _0','type of injury_0',
                            'suicide attempt']
feature_12_pow3 = ['HR','RR','SPO2 ','awareness_0','awareness_8','EMS','BP_1','BP_2',
                              'T','main symptoms_37','pain','pregnat']
feature_reg_13 = ['awareness_0', 'awareness_8', 'RR', 'SPO2 ', 'HR', 'BP_1', 'BP_2',
                              'T', 'Typ_size', 'Typ_fall_height', 'when', 'pain', 'pregnat']
feature_31_pow2 = ['HR','RR','SPO2 ','awareness_0','awareness_8','EMS','BP_1','BP_2',
                             'T','main symptoms_37','sex','pain','pregnat',
                              'awareness_1','main symptoms_0','additional information_0',
                              'place _0','place _4','place _6','place _8','place _10','place _22',
                              'old','dehydration_0', 'dehydration_1','breath_0','skin_9',
                              'medical history_0','Typ_size','when','intubation']
features_corr_14 =['awareness_8','HR','SPO2 ','RR','awareness_0','EMS',
                  'place _8','awareness_1','breath_0','main symptoms_0',
                   'skin_9','additional information_0','main symptoms_37',
                   'main symptoms_39']
feature_forest_45 = ['HR','awareness_8','awareness_0','SPO2 ','RR','BP_2','BP_1',
                     'old','pain','place _0','T','awareness_1','type of injury_0',
                     'main symptoms_0','EMS','when','place _1','dehydration_0',
                     'place _2', 'place _6', 'additional information_0','medical history_0',
                     'place _8','sex','Typ_size','awareness_9','main symptoms_37','pregnat',
                     'awareness_16','appetite','place _22', 'breath_0','place _7','dehydration_1',
                     'type of injury_3', 'skin_0','skin_9','place _9', 'fever','place _4',
                     'main symptoms_40', 'skin_13','intubation', 'place _10','medical history_22'
                    ]
feature_forest_45_add_2 = ['HR','awareness_8','awareness_0','SPO2 ','RR','BP_2','BP_1',
                     'old','pain','place _0','T','awareness_1','type of injury_0',
                     'main symptoms_0','EMS','when','place _1','dehydration_0',
                     'place _2', 'place _6', 'additional information_0','medical history_0',
                     'place _8','sex','Typ_size','awareness_9','main symptoms_37','pregnat',
                     'awareness_16','appetite','place _22', 'breath_0','place _7','dehydration_1',
                     'type of injury_3', 'skin_0','skin_9','place _9', 'fever','place _4',
                     'main symptoms_40', 'skin_13','intubation', 'place _10','medical history_22',
                    'skin_2','skin_3','appetite','dehydration_2',
                    ]
feature_tree_28=['awareness_8','BP_1','HR','old','BP_2','pain','EMS','SPO2 ',
                 'awareness_0','place _0','main symptoms_37','RR','dehydration_0',
                 'place _5','place _22','type of injury_11','place _6', 'T','place _10',
                 'medical history_4', 'main symptoms_4','additional information_2',
                 'place _4','dehydration_1','intubation','main symptoms_36','pregnat',
                 'medical history_6']    


# In[249]:


clean_data['old'].where(clean_data['old']<300).dropna()


# In[250]:


def select_features(features):
    data = finish_dataFrame[features]
    X =data.loc[:,features].values
    return X


# In[251]:


from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, Normalizer, Binarizer


# In[252]:


def scal_min_max(train_X,test_X):
    min_max_scaler = MinMaxScaler()
    x_train_scaled = min_max_scaler.fit_transform(train_X)
    x_test_scaled = min_max_scaler.fit_transform(test_X)

    return x_train_scaled, x_test_scaled


# In[253]:


from sklearn.preprocessing import StandardScaler

def standard(X):
    scaler = StandardScaler()
    scaler.fit(X)
    X_scaled = scaler.transform(X)
    return X_scaled


# In[254]:


X = select_features(feature_forest_45)
X_stand = standard(X)
train_X_min_max, test_X_min_max =scal_min_max(train_X,test_X)


# In[255]:


train_X, test_X, train_y, test_y,indices_train,indices_test = train_test_split(X_stand,
                                                                               y,
                                                                               finish_dataFrame.index,
                                                                               test_size=0.25,
                                                                               random_state=42)


# In[222]:


X = select_features(features_243)
X_stand = standard(X)

train_X, test_X, train_y, test_y = train_test_split(X_stand, y, test_size=0.25, random_state=42)
train_X_min_max, test_X_min_max =scal_min_max(train_X,test_X)


# In[171]:


pca = PCA(0.60)
#pca=PCA(n_components = 3)
train_X_2 = train_X_1
pca.fit(train_X_2)

train_X= pca.transform(train_X_2)
test_X= pca.transform(test_X_1)
print(pca.n_components_)


# In[256]:


len(train_X)


# In[42]:


from sklearn import svm
from sklearn import tree
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# In[43]:


gnb_0 = MultinomialNB()
tree_0 = tree.DecisionTreeClassifier(random_state = 8)
rf_0 = RandomForestClassifier(random_state = 8)
knnc_0 =KNeighborsClassifier()
lr_0 = LogisticRegression(random_state = 8)
svc_0 =svm.SVC(random_state=8)

lista_modeli = {'Naiwny Bayes':gnb_0, 'Drzewo decyzyjne':tree_0,
                'Lasy losowe':rf_0,'KNN':knnc_0,'Regresja logiczna':lr_0,
                'SVM':svc_0}


# In[44]:


#Random Forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 1000, num = 5)]

# max_features
max_features = ['auto', 'sqrt']

# max_depth
max_depth = [int(x) for x in np.linspace(20, 100, num = 5)]
max_depth.append(None)

# min_samples_split
min_samples_split = [2, 5, 10]

# min_samples_leaf
min_samples_leaf = [1, 2, 4]

# bootstrap
bootstrap = [True, False]

# Create the random grid
random_grid_lr = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap,
                "criterion": ["gini", "entropy"]}


n_estimators = [int(x) for x in np.linspace(start = 200, stop = 1000, num = 5)]

# max_features
max_features = ['auto', 'sqrt']

# max_depth
max_depth = [int(x) for x in np.linspace(20, 100, num = 5)]
max_depth.append(None)

# min_samples_split
min_samples_split = [2, 5, 10]

# min_samples_leaf
min_samples_leaf = [1, 2, 4]

# bootstrap
bootstrap = [True, False]

# Create the random grid
random_grid_tree = {
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}
# C
C = [.0001, .001, .01]

# gamma
gamma = [.0001, .001, .01, .1, 1, 10, 100]

# degree
degree = [1, 2, 3, 4, 5]

# kernel
kernel = ['linear', 'rbf', 'poly']

# probability
probability = [True]

# Create the random grid
random_grid_svm = {'C': C,
              'kernel': kernel,
              'gamma': gamma,
              'degree': degree,
              'probability': probability
             }

n_neighbors = [int(x) for x in np.linspace(start = 1, stop = 20, num = 20)]

param_grid_knn = {'n_neighbors': n_neighbors}

alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
random_grid_nb = {'alpha': alphas
               #'class_prior ' :lista_val
              }

# C
C = [float(x) for x in np.linspace(start = 0.1, stop = 1, num = 10)]

# multi_class
multi_class = ['multinomial']

# solver
solver = ['newton-cg', 'sag', 'saga', 'lbfgs']
 
# class_weight
class_weight = ['balanced', None]

# penalty
penalty = ['l2']

# Create the random grid
random_grid_lg = {'C': C,
               'multi_class': multi_class,
               'solver': solver,
               'class_weight': class_weight,
               'penalty': penalty}



param = {'Naiwny Bayes':random_grid_nb,'Drzewo decyzyjne':random_grid_tree,
                'Lasy losowe':random_grid_lr,'KNN':param_grid_knn,'Regresja logiczna':random_grid_lg,
                'SVM':random_grid_svm}


# In[45]:


def randomSearch(value_mode,name_model):
     # Zdefiniowanie wyszukiwania losowego
    random_search = RandomizedSearchCV(estimator=value_mode,
                                       param_distributions= param[name_model],
                                       n_iter=50,
                                       scoring='accuracy',
                                       cv=3, 
                                       verbose=1, 
                                       random_state=8)
    return random_search


# In[46]:


lista_modeli


# In[47]:


test_X.shape[1]


# In[257]:


def randomSearch(value_mode,name_model):
     # Zdefiniowanie wyszukiwania losowego
    random_search = RandomizedSearchCV(estimator=value_mode,
                                       param_distributions= param[name_model],
                                       n_iter=50,
                                       scoring='accuracy',
                                       cv=5, 
                                       verbose=1, 
                                       random_state=8)
    return random_search
wyniki_list=[]
best_estymator={}
for name_model, value_mode in lista_modeli.items():
    if name_model == 'Naiwny Bayes':
            random_search =randomSearch(value_mode,name_model)
            random_search.fit(train_X_min_max, train_y)
            best_estimator = random_search.best_estimator_
            best_estymator[name_model]=random_search.best_estimator_
            wyniki={'Model': name_model,
            'Najleprzy estymator':random_search.best_estimator_,
            'liczba cech':test_X.shape[1],
            'Wartość średnia dokładności zbioru uczącego':float(random_search.best_score_)*100}
           
            wyniki_list.append(wyniki)

    else:
            random_search =randomSearch(value_mode,name_model)
            random_search.fit(train_X, train_y)
            best_estymator[name_model]= random_search.best_estimator_
            
            wyniki={'Model': name_model,
            'Najleprzy estymator':random_search.best_estimator_,
            'liczba cech':test_X.shape[1],
            'Wartość średnia dokładności zbioru uczącego':float(random_search.best_score_)*100
            }
            wyniki_list.append(wyniki)


# In[258]:


wyniki_list[2][ 'Najleprzy estymator']


# In[259]:


best_estymator['Lasy losowe']


# In[260]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def add_Data():
    data_list =[]
    for x in range(len(wyniki_list)):
        if wyniki_list[x]['Model']=='Lasy losowe':
            data =pd.DataFrame([wyniki_list[x]], index=[0])
            #list_str = wyniki_list
            #list_str[x]['Najleprzy estymator']= "RandomForestClassifier(bootstrap=False, max_depth=60, n_estimators=200, random_state=8)"
            #print( list_str[x])
            #data =pd.DataFrame([list_str[x]], index=[0])
            data_list.append(data)
        else:
            data =pd.DataFrame([wyniki_list[x]], index=[0])
            data_list.append(data)
            
        
    result = pd.concat(data_list)
    return result 

def add_recall_pre(result):
    lista_acc=[]
    list_recall=[]
    list_pre=[]
    list_matrix =[]
    predict_list = []
    for name_model, value_mode in lista_modeli.items():
        if name_model == 'Naiwny Bayes':
            model =value_mode
            model.fit(train_X_min_max, train_y)
            predict = model.predict(test_X_min_max)
            acc = float(accuracy_score(test_y,predict)) *100
            list_pre.append(precision_score(test_y, predict, average=None))
            list_recall.append(recall_score(test_y, predict, average=None))
            list_matrix.append(confusion_matrix(test_y, predict))
            predict_list.append(predict)

            lista_acc.append(acc)
        else:
            model =value_mode
            model.fit(train_X, train_y)
            
            predict = model.predict(test_X)
            predict_list.append(predict)
            acc = float(accuracy_score(test_y,predict)) *100
            precision_score(test_y, predict, average=None)
            recall_score(test_y, predict, average=None)
            list_pre.append(precision_score(test_y, predict, average=None))
            list_recall.append(recall_score(test_y, predict, average=None))
            list_matrix.append(confusion_matrix(test_y, predict))
            lista_acc.append(acc)
                               
    result['Dokłdność zbioru testowego']=lista_acc
    result['Czułość']=list_recall
    result['Swoistość']=list_pre
    result['Macierz pomyłek']=list_matrix
    result['predykcja']= predict_list
    
    
    
result = add_Data()
add_recall_pre(result)

sort_result = result.sort_values('Dokłdność zbioru testowego', ascending=False)
index_result = sort_result.reset_index().drop('index', axis=1)


# In[ ]:





# In[261]:


conf_matrix = index_result['Macierz pomyłek'].loc[0]
plt.figure(figsize=(6,5))
sns.heatmap(conf_matrix, 
            annot=True,
            xticklabels=[1,2,3,4,5], 
            yticklabels=[1,2,3,4,5],
            cmap="Blues")
plt.ylabel('przewidywane')
plt.xlabel('rzeczywiste')
plt.title('Macierz pomyłek')
plt.show()


# In[232]:


index_result


# In[55]:


from IPython.display import display, HTML
display(HTML(index_result.to_html()))


# In[ ]:





# In[56]:


index_result['Macierz pomyłek'].loc[0]


# In[ ]:





# In[150]:


from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix 


kf = KFold(n_splits=3)



#for train, test in kf.split(X):
#    print("%s %s" % (train_X, test_X))
pre = index_result['predykcja'][0]
plt.figure(figsize=(6,5))
sns.heatmap(confusion_matrix(test_y, pre), 
            annot=True,
            xticklabels=[1,2,3,4,5], 
            yticklabels=[1,2,3,4,5],
            cmap="Blues")
plt.ylabel('przewidywane')
plt.xlabel('rzeczywiste')
plt.title('Macierz pomyłek')
plt.show()


# In[58]:


model = best_estymator['Lasy losowe']
model.fit(test_X,test_y)


# In[95]:


best_estymator['Lasy losowe'].fit(train_X, train_y)
pred = pre
print(classification_report(test_y,pred))


# In[60]:


best_estymator['Lasy losowe']


# In[139]:


from sklearn.model_selection import cross_val_predict, KFold
kfoldd = model_selection.KFold(n_splits=5, random_state=None)
model = best_estymator['Lasy losowe']
kf_5 =model_selection.KFold(n_splits=5, random_state=None)

prediction_cv = cross_val_predict(model, train_X, train_y, cv=kf_5)

print(prediction_cv)
print(train_y)

print(confusion_matrix(train_y, prediction_cv))
print(recall_score(train_y, prediction_cv, average=None))
print(precision_score(train_y, prediction_cv, average=None))


# In[233]:


plt.figure(figsize=(6,5))
sns.heatmap(confusion_matrix(train_y, prediction_cv), 
            annot=True,
            xticklabels=[1,2,3,4,5], 
            yticklabels=[1,2,3,4,5],
            cmap="Blues")
plt.ylabel('przewidywane')
plt.xlabel('rzeczywiste')
plt.title('Macierz pomyłek')
plt.show()


# In[141]:


misclassified_8 = np.where(train_y != prediction_cv)
print(misclassified_6)


# In[ ]:


rl= best_estymator['linear_regression']
clf = rl.fit(train_X,train_y)
cm_holder = []
kfold = model_selection.KFold(n_splits=10, random_state=42)
for train_index, test_index in kfold.split(train_X):
    X_train, X_test = train_X[train_index], train_X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print(y_test)
   
    rl.fit(X_train, y_train)
    print(rl.predict(X_test))
    print(precision_score(y_test, rl.predict(X_test), average=None))
    cm_holder.append(confusion_matrix(y_test, rl.predict(X_test)))


# In[ ]:





# In[ ]:





# In[ ]:


from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
scoring = {'accuracy' : make_scorer(accuracy_score), 
       'precision' : make_scorer(precision_score, average = 'micro'),
       'recall' : make_scorer(recall_score, average = 'micro'), 
       'f1_score' : make_scorer(f1_score, average = 'micro')}
model = best_estymator['Decidion_Tree']
model =model.fit(train_X, train_y)
kfold = model_selection.KFold(n_splits=3, random_state=42)
results = model_selection.cross_validate(estimator=model,
                                          X=train_X,
                                          y=train_y,
                                          cv=kfold,
                                          scoring=scoring)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


from sklearn.ensemble import RandomForestClassifier
#model = RandomForestClassifier(n_estimators=100, max_depth=5)
import pydotplus # To create our Decision Tree Graph
from IPython.display import Image  
# Train
model =tree.DecisionTreeClassifier(
max_depth=3, 
max_features='sqrt', 
random_state=8).fit(train_X, y_train)
# Extract single tree
estimator = model
estimator
dot_data = tree.export_graphviz(estimator, out_file=None, feature_names=feature_forest_45, 
                                class_names=['1', '2','3','4','5'],
                                rounded=True, filled=True,
                               special_characters=True) #Gini decides which attribute/feature should be placed at the root node, which features will act as internal nodes or leaf nodes
#Create Graph from DOT data
graph = pydotplus.graph_from_dot_data(dot_data)

# Show graph
Image(graph.create_png())


# In[ ]:


pred = best_estymator['Lasy losowe'].predict(test_X)


# In[ ]:


pred


# In[92]:


pred[46]


# In[ ]:


test_X[0]


# In[ ]:


test_y = np.asarray(test_y)


# In[115]:



misclassified = np.where(test_y != index_result['predykcja'][0])
indices_test[misclassified]


# In[116]:


misclassified


# In[ ]:





# In[91]:





# In[118]:


indices_test[misclassified]


# In[94]:


clean_data.loc[119]


# In[117]:


finish_dataFrame.loc[[ 0,  3,  8, 13, 14, 27, 37, 46],['HR','awareness_8','awareness_0','SPO2 ','RR','BP_2','BP_1',
                     'old','pain','place _0','T','awareness_1','type of injury_0',
                     'main symptoms_0','EMS','when','place _1','class']]


# In[234]:


clean_data.loc[[66, 193, 154, 137, 146, 225, 125, 147, 119, 112, 150, 55, 213,
            117],['HR','RR','SPO2 ','awareness','EMS','class']]


# In[451]:


misclassified_oki = np.where(test_y != best_estymator['Lasy losowe'].predict(test_X))


# In[452]:


indices_test[misclassified_oki]


# In[453]:


clean_data.loc[indices_test[misclassified_oki]].where(clean_data['class']==1).dropna()


# In[ ]:





# In[319]:





# In[ ]:





# In[ ]:




