#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn import model_selection
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error
from math import sqrt
get_ipython().run_line_magic('matplotlib', 'inline')

#wywołnie funkcji

def clean_dataFrame(data_cop):
    clean_vital_signal(data_cop)
    replace_palp(data_cop)
    drop_add_col(data_cop,16,'additional information')
    add_words_forget(data_cop)
    change_words(data_cop)

    drop_col_class_two(data_cop)
    #print(data_cop.head())
    tag_main_symptoms(data_cop,sym_words)
    tag_pain(data_cop)
    tag_breath(data_cop, breath_words)
    
    tag_fever(data_cop)
    tag_skin(data_cop,skin_word)
    tag_typs_injury(data_cop,types_words)
    tag_dehydration(data_cop,deh_words)
    tag_appetite(data_cop)

    tag_his_medical(data_cop,med_his_words)
    tag_additional_inf(data_cop,add_inf_words)
    add_val_awareness_4_5(data_cop)
    add_no_if_nan(data_cop)
    tag_pregnat(data_cop)
    tag_awareness(data_cop,aw_words)

    tag_place(data_cop,body)
    clean_place(data_cop)

    add_col_typ_size(data_cop)
    add_col_typ_fall_height(data_cop)


    clean_col_stop_words(data_cop)
    tag_suicide_at(data_cop)
    data_cop.loc[219,'suicide attempt']=0
    uniform_value_pregnat(data_cop)
    uniform_value_old(data_cop)
    add_value_vital_signal_and_drop_col(data_cop)  
    tag_intubation_EMS_bleeding_alcohol_con_CPR_sex(data_cop)
    change_val_SPO2(data_cop)
    uniform_value_when(data_cop)
    tag_defibrillated(data_cop)
    BP_1,BP_2 = separate_col_BP(data_cop)
    drop_add_BP(data_cop,BP_1,BP_2)
    for x in ['BP_1','BP_2','RR','HR','T']:
        data_cop[x] = pd.to_numeric(data_cop[x])
    for i in ['HR','RR','SPO2 ','BP_1','BP_2']:
        interquartile (data_cop,i)
    
    uzupelnij_HR_regresja(data_cop,'HR','RR')  
    interquartile (data_cop ,'RR')
    #uzupelnij_HR_regresja(data_cop,'RR','HR')
    interquartile (data_cop ,'SPO2 ')
    #uzupelnij_HR_regresja(data_cop,'RR','SPO2 ')
    
    add_vital_signal_no_breath(data_cop)
    add_median_vital_signal(data_cop)
    #add_median_vital_signal_global(data_cop)
    data_cop['appetite'] = pd.to_numeric(data_cop['appetite'])
    data_cop['pregnat'] = pd.to_numeric(data_cop['pregnat'])
    data_cop['suicide attempt'] = pd.to_numeric(data_cop['suicide attempt'])
    data_cop['Typ_size'] = pd.to_numeric(data_cop['Typ_size'])


    return data_cop


def uzupelnij_HR_regresja(data_cop,X,Y):
    regr = reg_vital_signal(data_cop,[X],[Y])
    for x,y in enumerate(data_cop[Y]):
        if 'nan' in str(y) and str(data_cop.loc[x,X])!=str(np.nan):
            #print('indeks: ',x,', wartość',Y,':',y,',wartość',X,':',data_cop.loc[x,X])
            data_cop.loc[x,Y]= match_value(regr, [float(data_cop.loc[x,X])])
            
def reg_vital_signal(data_cop,cechy,cecha):
    par = clean_data_to_regresion(data_cop,cechy[0],cecha[0])
    X=par[cechy].values.reshape(-1,1)
    
    y=par[cecha]
    splits = model_selection.train_test_split(X, y, test_size=0.20, random_state=0)
    X_train, X_test, Y_train, Y_test = splits
    regr = linear_model.LinearRegression()
    # Train the model using the training sets
    regr.fit(X_train, Y_train)
    # Predict values
    Y_predicted = regr.predict(X_test)
    # The coefficients
    #print('Coefficients: \n', regr.coef_)
    #intercept_ 
    #print('intercept_: \n', regr.intercept_)   
    #print ( 'Regresja liniowa R do kwadratu: % .4f '  %  regr . score ( X_test ,  Y_test ))
    rmse = sqrt(mean_squared_error(Y_test, Y_predicted))
    #print('X_test',len(X_test), 'Y_test: ',len(Y_test))
    #print('RMSE:', rmse )
    #plt.scatter(X_test, Y_test,  color='black')
    #plt.plot(X_test, Y_predicted, color='blue', linewidth=3)
    
    return regr


def clean_data_to_regresion(data_cop,par_1,par_2):
    data_vital = data_cop[['RR', 'SPO2 ', 'HR']]
    drop_RR_HR = data_vital.dropna(subset = [par_1,par_2])
    #print(len(drop_RR_HR),par_1,par_2)
    #drop_RR_HR = data_vital[['RR','HR','SPO2 ','old','class','BP_1','BP_2']]
    drop_RR_HR.reset_index(drop=True, inplace=True)
    return drop_RR_HR

def match_value(regr, values):
    features = values
    x_pred = np.array(values)
    x_pred = x_pred.reshape(-1, len(features))
    #print(regr.predict(x_pred))
    
    return int(regr.predict(x_pred))

def interquartile ( data_cop ,  col):   
    for nr_class in range(1,6):
        Q1 = data_cop[col].where(data_cop['class']==nr_class).quantile(0.25)
        Q3 = data_cop[col].where(data_cop['class']==nr_class).quantile(0.75)
        Q2 = data_cop[col].where(data_cop['class']==nr_class).median()
        IQR = Q3 - Q1
        low = Q1 - 1.5 * IQR
        high = Q3 + 1.5 * IQR
        index_low = data_cop . loc [( data_cop [col ].where(data_cop['class']==nr_class)  <  low ) ].index
        index_high = data_cop . loc [( data_cop [col ].where(data_cop['class']==nr_class)  >  high )].index
        #print('wartość Q1 dla cechy: ',col,'oraz klasy: ',nr_class,'wynosi:', Q1)
        #print('próbki odstające(poniżej poziomu min):')
        for x in index_low:
            #print(x,clean_data[col].loc[x])
            data_cop[col].loc[x] = Q1
        #print('wartość Q3 dla cechy: ',col,'oraz klasy: ',nr_class,'wynosi:', Q3)
        #print('próbki odstające(powyżej poziomu max):')
        for x in index_high:
            #print(x,clean_data[col].loc[x])
            data_cop[col].loc[x] = Q3
           

            

def clean_vital_signal(data_cop):
    INDEX_vital =['BP', 'HR', 'RR', 'T']
    for x,y in enumerate(data_cop['vital signal']):
        if '78' in str(y):
            data_cop.loc[x,'vital signal'] = np.NaN
            a =str(y).split('xxx')
            #print(x)
            for index, val in enumerate(INDEX_vital):
                data_cop.loc[x,val] = a[index].replace(val,'',True)
                if 'F' in str(y):
                    data_cop.loc[x,val] = data_cop.loc[x,val].replace('F','',True)


def list_median_vital(data_cop):
    list_val_med=[]
    for x in range(1,6):
        a,b,c,d,e,f =data_cop[['RR','HR','BP_1','BP_2','T','SPO2 ']].where(data_cop['class']==x).median()
        per=a,b,c,d,e,f
        list_val_med.append(per)
    return list_val_med
def list_median_vital_global(data_cop):
    list_val_med=[]
    a,b,c,d,e,f =data_cop[['RR','HR','BP_1','BP_2','T','SPO2 ']].median()
    per=a,b,c,d,e,f
    list_val_med.append(per)
    return list_val_med

def add_vital_signal_no_breath(data_cop):
    for x,y in enumerate(data_cop['breath']):
        if '3' in str(data_cop.loc[x,'breath']):
            data_cop.loc[x,'RR']=0
            data_cop.loc[x,'HR']=30
            data_cop.loc[x,'BP_1']=72
            data_cop.loc[x,'BP_2']= 53
            data_cop.loc[x,'SPO2 ']= 87
            data_cop.loc[x,'T']=98.0 
        if '10' in str(data_cop.loc[x,'breath']):
             data_cop.loc[x,'RR']=50
        if str(data_copi.loc[x,'defibrillated ']) != 'nan':
            data_cop.loc[x,'RR']=0
            data_cop.loc[x,'HR']=30
            data_cop.loc[x,'BP_1']=72
            data_cop.loc[x,'BP_2'] =53
            data_cop.loc[x,'SPO2 ']= 87
            data_cop.loc[x,'T']=98.0 

skin_word ={np.NaN:0,'rash':1,'dry':2,'cracked':3,'swelling':4,'moist lesions':5,'tender':6,'warm':7,'cool':8,
            'cold':8,'diaphoretic':9, 'purpuric lesions':10,'petechial':11,'little pinpoint purplish':11,'mottle':12,'red':13,
            'blood':13,'swollen':14,'pus':15,'pale':16,'clammy':17,'small paronychia':18,'tearing':19,'itchy':20,
            'very mucous membranes':21,'bruising':22,'crusty':23,'unable to palpate a pulse':24,'flushed':25}

def tag_skin(data_cop,skin_word):
    match_words(data_cop,'skin',skin_word)

    data_cop['skin'] = data_cop['skin'].str.replace('12d','12')
    words={'dened':'','ness':'','new':'','area':'','across':'','to touch':'','sandpaper':'','very':'','is oozing from around the site':''}
    match_words(data_cop,'skin',words)
    data_cop['skin'] = data_cop['skin'].str.replace(',','')

def replace_palp(data_cop):
    data_cop['BP']= data_cop['BP'].str.replace('palp','60',True)
    data_cop['BP']= data_cop['BP'].str.replace('pal','60',True)

def tag_suicide_at(data_cop):
    data_cop['suicide attempt'] = data_cop['suicide attempt'].map({'no':0,'drug overdose':1,'selfinflicted':2})

def drop_add_col(data_cop,index_col, name_where_add):
    column_names = data_cop.columns.tolist()
    for index, val_activ in enumerate(data_cop[column_names[index_col]]):
        if data_cop.loc[index, column_names[index_col]] == 'yes':
            data_cop.loc[index, name_where_add] = 'physical activity'

    data_cop.drop(column_names[index_col],1, inplace =True)

        
def add_words_forget(data_cop):
    data_cop.loc[10,'place ']= 'vagina'
    data_cop.loc[10,'skin'] = 'pale'
    data_cop.loc[10,'awareness'] = 'before I pass out'
    data_cop.loc[10,'main symptoms']='passing clots the size of oranges'
    data_cop.loc[107,'awareness'] = 'anxious xxx despairing'
    data_cop.loc[185,'pregnat'] = 'no'
    data_cop.loc[188,'pregnat'] = 'no'
    
    data_cop.loc[41,'additional information']='requiring insulin occlusion'
    data_cop.loc[41,'skin'] =data_cop.loc[41,'skin'] + 'unable to palpate a pulse'
    data_cop.loc[142,'place ']='lungs'
    data_cop.loc[198,'place ']='foot'
    data_cop.loc[171,'type of injury']='maybe bitten'


def replace_all(data_cop,col,index,d):
    for a,b in d.items():
        data_cop.loc[index,col] = str(data_cop.loc[index,col]).lower().replace(str(a),str(b))
        #print(index,data_cop.loc[index,col])


def match_words(data_cop,text, words):
    for x,txt in enumerate(data_cop[text]):
        replace_all(data_cop,text, x, words)
        if '' == str(data_cop.loc[x,text]).strip():
            #print(index,data_cop.loc[x,txt])
            data_cop.loc[x,text]=np.NaN

def add_remove_words(data_cop,col_remove, col_add, words):
    for word in words:
        for x,y in enumerate(data_cop[col_remove]):
            if word in str(y):
                if str(word) == str(y).strip():
                    data_cop.loc[x,col_remove]= np.NaN
                else:
                    #print(x,y)
                    data_cop.loc[x,col_remove]= data_cop.loc[x,col_remove].replace(word,'')
                    if 'xxx' == str(data_cop.loc[x,col_remove]).strip() or 'and'==str(data_cop.loc[x,col_remove]).strip():
                        data_cop.loc[x,col_remove]= np.NaN
                if 'nan' in str(data_cop.loc[x,col_add]):
                    if str(col_add)=='pain':
                        if 'severe' in str(word) or 'hard' in str(word):
                            data_cop.loc[x,col_add]= 8
                        else:
                            data_cop.loc[x,col_add]= 6
                    else:
                        data_cop.loc[x,col_add]= word
                else:
                    if str(col_add)!='pain':
                         data_cop.loc[x,col_add]= data_cop.loc[x,col_add] + " "+ str(word)
                        
def drop_col_class_two(data_cop):
    count_col_not_null = {x:data_cop.loc[x].count() for x,y in enumerate(data_cop['class']) if '2'==str(y)}
    sort_orders = sorted(count_col_not_null.items(), key=lambda x: x[1], reverse=False)
    index_col_to_drop=[x for x,y in sort_orders[:20] ]
    #print(sort_orders[:20])
    data_cop.drop(index_col_to_drop, inplace=True)
    data_cop.reset_index(drop=True, inplace=True)

def change_words(data_cop):
    skin_words=['new reddened area','tenderness','itchy','crusty','tearing','reddened','redness','swollen','swelling',
                'small paronychia','bruising']
    type_words=['3-centimeter laceration','broken','right leg is shortened and externally rotated','no obvious deformity',
               'obviously deformed','obvious deformity','movement',"can't move",'ragged','stepped on a rusty nail',
               'injured','stiff','abrasion','refusing to move','twisted','dislocation','flaccid']
    awareness_words=['sensation decreased','exhausted','feeling run down','dizzy']
    place_words=['throat','tooth',' abdominal']
    add_inf_words=['suspicion positive streptococci','colitis is acting up','medications not work',
                   'large hematoma around the wound',"EKG - anterior lateral ischemic changes",
                  'passing clots the size of oranges','irregular HR','obvious chronic obstructive pulmonary disease']

    add_remove_words(data_cop,'main symptoms','type of injury',type_words)
    add_remove_words(data_cop,'main symptoms','skin',skin_words)
    add_remove_words(data_cop,'main symptoms','awareness',awareness_words)
    add_remove_words(data_cop,'main symptoms','place ',place_words)
    add_remove_words(data_cop,'place ','type of injury',['2-centimeter laceration'])
    add_remove_words(data_cop,'main symptoms','additional information',add_inf_words)
    add_remove_words(data_cop,'awareness','main symptoms',['sudden slurred speech','slurred speech'])
    add_remove_words(data_cop,'type of injury','awareness',['flaccid'])
    add_remove_words(data_cop,'type of injury','place ',['right hip','right leg','face','left arm'])
    add_remove_words(data_cop,'skin','place ', ['chest','lips','mucous membranes','left arm','face','right hip"'])
    add_remove_words(data_cop,'type of injury','main symptoms',['something in'])

def tag_pain(data_cop):
    sym_nan_words={np.NaN:0}
    pain =['hurts','hurt','hard','hart','ache','sore','painful','sore','severe pain','severe','pain','xxx   xxx']
    add_remove_words(data_cop,'main symptoms','pain',pain)
    data_cop['pain']=data_cop['pain'].replace(20,10)
    match_words(data_cop,'main symptoms',sym_nan_words)

def tem_F(data_cop):
    for x,y in enumerate(data_cop['T']):
        if 'C' in str(y):
            a =float(data_cop.loc[x,'T'].replace('C',''))
            data_cop.loc[x,'T'] =(a*1.8)+32
med_his_words={np.NaN:0,'htn':1,'medication for blood pressure':1,'asthma':2,'asthhma':2,'2tic':2,'uses several metered-dose inhalers':2,
               'migraine':3,'migreny':3,'3s':3,'diabetic':4,
               'diabetes':4,'type 2 4':4,'insulin':4,'6 weeks post laparoscopic gastric bypass':5,
              'metastatic ovarian cancer':6,'poison ivy':7,'arthritis':8,'chronic renal failure':9,'regular kidney dialysis':9,
              'aphasic':10, 'massive 11 3 years ago':11,'massive 11':11,'stroke':11,'suffered a massive 11':11,
               'therapeutic abortion':13,'cardiac':14,'14s':14,'heart attack':14,'infarction':14,'atrial fibrillations':14,'atrial fibrillation':14,
               'multiple medications':15,'medications':15,'gastroesophageal reflux disease':16,'medications include fioricet':17,  'include fioricet':17,
               'allergic to penicillin':18,'severe shellfish allergy':18, 'warfarin':19,
               'amiodarone':20,'chronic obstructive pulmonary disease':21,'healthy':22,'psoriasis':23,
               'colitis':24,'intubations':25,'intubated':25,'clots in my legs':26, 'high cholesterol':27,
               'epipen':2,'lymphoma':28,'lung cancer':28,'metastatic breast cancer':28,'completed chemotherapy': 35,
               'multiple sclerosis':29,'chemotherapy':30,'sickle cell disease':31,'no tetanus immunization':32,
               'last tetanus immunization was 10 years ago':32,'aspirin':33,'ractured':34,'kidney stones':35,
               'previous ectopic preg0cy':39,
               'antibiotics for 5 days for mastitis':40,'ibuprofen':41,'frequent ear infections':42,'ear infections':42,'no allergies':43,
                'no meds':44,'no medications':44,'immunizations are up to date':45,'no 15':44}

def tag_his_medical(data_cop,med_his_words):


    for x,typ in enumerate(data_cop['medical history']):
        is_list =0
        if 'diuretic' in str(typ):
            data_cop.loc[x,'medical history'] = data_cop.loc[x,'medical history'].replace('diuretic','12 15')
        if 'knee replacement' in str(typ):
            data_cop.loc[x,'medical history'] = data_cop.loc[x,'medical history'].replace('knee replacement','36 37')
        if 'gastric bypass' in str(typ):
            data_cop.loc[x,'medical history'] = data_cop.loc[x,'medical history'].replace('gastric bypass','36 38')
        if  'pill to thin his blood' in str(typ):
            data_cop.loc[x,'medical history'] = data_cop.loc[x,'medical history'].replace( 'pill to thin his blood','15 26')
        if 1:
            replace_all(data_cop,'medical history',x, med_his_words)



    match_words(data_cop,'medical history',{'2 weeks ago':'','late-stage non-hodgkinős':'','wrist':'','3 months ago':'','3 weeks ago':'','3 years ago':'','6 weeks post laparoscopic':''})
    data_cop['medical history'] = data_cop['medical history'].str.replace(',','').str.replace('massive','')

body = {np.NaN:0,'head':1,'forehead':1,'face':1,'cheeks':1,'facial':1,'jaw':1,'nose':1,'periorbital area':1,
        'abdominal':2,'belly':2,'chest':3,'breast':3,'lungs':3, 'throat':4,'neck':4, 'eye':5,'arm':6,'hand':6,
        'houlder':6,'wrist':6,'leg':7,'hip':7,'feet':7,'calf':7,'ankle':7,'legs':7,'ankles':7,
        'thigh':7,'knee':7,'ear':8, 'ears':8,'urinary tract':9,'testicle':9,'vaginal':9,'vagina':9,'groin':9,'crotch':9,
        'penis':9,'tooth':10,'clavicle':11,'lip':12,'Lips':12,'toenails':13,'rectum':14,'costovertebral angle':19, 
        'mucous membranes':21,'second finger':22,'thumb':22, 'finger':22,'foot':23,'toes':23,'all over':24,'30 %  body':24}

           
def add_median_vital_signal(data_cop):
    lista_parameter=['RR','HR','BP_1','BP_2','T','SPO2 ']
    for x,y in enumerate(data_cop['class']):
        for nr_class in range(1,6):
            if y==nr_class:
                for index,parameter in enumerate(lista_parameter):
                    if str(data_cop.loc[x,parameter])== 'nan':
                        #print(x,y,nr_class,index,parameter,list_val_med[nr_class-1][index])
                        data_cop.loc[x,parameter]=list_median_vital(data_cop)[nr_class-1][index]
                        
def add_median_vital_signal_global(data_cop):
    for x,y in enumerate(data_cop['class']):
        lista_parameter=['RR','HR','BP_1','BP_2','T','SPO2 ']
        list_val_med = list_median_vital_global(data_cop)
        for index,parameter in enumerate(lista_parameter):
             if str(data_cop.loc[x,parameter])== 'nan':
                #print(x,y,index,parameter)
                data_cop.loc[x,parameter]=list_val_med[0][index]

                        
def tag_place(data_cop,body):

    for x,typ in enumerate(data_cop['place ']):
        is_list =0
        if 'left lower quadrant abdominal' in str(typ):
            is_list =1
            data_cop.loc[x,'place '] = data_cop.loc[x,'place '].replace('left lower quadrant abdominal','17 2')
        if 'left lower-quadrant abdominal' in str(typ):
            is_list =1
            data_cop.loc[x,'place '] = data_cop.loc[x,'place '].replace('left lower-quadrant abdominal','17 2')
        if 'stomach' in str(typ):
            is_list =1
            data_cop.loc[x,'place '] = data_cop.loc[x,'place '].replace('stomach','15 2')
        if 'right lower quadrant abdominal' in str(typ):
            is_list =1
            data_cop.loc[x,'place '] = data_cop.loc[x,'place '].replace('right lower quadrant abdominal','18 2')
        if 'left upper quadrant abdominal' in str(typ):
            is_list =1
            data_cop.loc[x,'place '] = data_cop.loc[x,'place '].replace('left upper quadrant abdominal','20 2')
        if 'mid-abdominal' in str(typ):
            is_list =1
            data_cop.loc[x,'place '] = data_cop.loc[x,'place '].replace('mid-abdominal','16')
        if 'low abdominal'  in str(typ):
            is_list =1
            data_cop.loc[x,'place '] = data_cop.loc[x,'place '].replace('low abdominal','17 18 2')
        if 'lower quadrant abdominal' in str(typ):
            is_list =1
            data_cop.loc[x,'place '] = data_cop.loc[x,'place '].replace('lower quadrant abdominal','17 18 2')
        if 'button' in str(typ):
            is_list =1
            data_cop.loc[x,'place '] = data_cop.loc[x,'place '].replace('button','16 2')
        if 'extremitie' in str(typ):
            is_list =1
            data_cop.loc[x,'place '] = data_cop.loc[x,'place '].replace('extremitie','6 7')
        if is_list == 0:
            replace_all( data_cop,'place ',x, body)

breath_words={np.nan:0, 'wheezing':1,'short of breath':2,'no':3,'respiratory arrest':3,'working hard':4,'hard time breathing':4,
              '3n-labored respirations':5,'spontaneously':6,'barely catch':7,'canőt breathe':7, 
              'respiratory distress':8,'9 after coughing':9,'shortness':9,'9 of breath':9, 'increased work of breathing':10,'increasing':11 }

def tag_breath(data_cop, breath_words):   
    match_words(data_cop,'breath',breath_words)
    match_words(data_cop,'breath',{'earlier':'','sudden onset of':'','moderate':'', 'increbodasing':'','after coughing':'','severe':''})

add_inf_words={np.NaN:0,'car accident':1,'vehicle collision':1,'physical activity':2,'accident in water':3,'pain medication not working':4,
           'acetaminophen but it didnőt help':4,'inhaler doesnőt seem to be helping':4,
            'medications not work':4,'changed medications':5,'i feel okay when i lie perfectly still':6,
            'morbidly obese':7,'driving in a car for 12 hours':8,
            'has been sitting in a car for the last 2 days':8,'did not go to dialysis':9,'colitis is acting up':10,'sibling positive strep culture':11,
            'suspicion positive streptococci':11,'large hematoma around the wound':12,
            'ekg - anterior lateral ischemic changes':13,'passing clots the size of oranges':14,'irregular hr':15,
            'obvious chronic obstructive pulmonary disease':16,'requiring insulin occlusion':17}

def tag_additional_inf(data_cop, add_inf_words):    
    add_remove_words(data_cop,'medical history','additional information',['did not go to dialysis'])
    match_words(data_cop,'additional information', add_inf_words)
def clean_place(data_cop):
    stop_words_place = ['left','right','low','lower','bilateral','second','side','of','his','shoulder','fore','xxx',
                       'anterior','s','area','fore','around','the','and','first','belly','sur1']
    pat = '|'.join(r"\b{}\b".format(x) for x in stop_words_place)
    data_cop['place '] = data_cop['place '].str.replace(pat, '')
    data_cop['place '] = data_cop['place '].str.replace('s','').str.replace('fore','').str.replace('thigh','7').str.replace(',','')

    data_cop['place '] = data_cop['place '].str.split()
sym_words={'cough':1,'hoarse':2,'rapid heart rate':3,'funny pressure feeling':4, 'discomfort':4,'sudden headache':5,
           'pressure':4,'sudden onset of a severe headache':5,'foley catheter came out':6,'replacement of her peg tube':7,
           'nausea':8,'fluid buildup':9, 'maroon-colored stool':10,'dark stools':10,'defibrillator gave me a shock':11,'anaphylactic shock':12,
           'strong allergic reaction':12,'tachycardic':13,'sudden onset of palpitations':14,'i couldnőt see about 5 minutes':15,
           'loss of vision':15,'alcohol delirium':16,'feeding tube fell out':17,'sexually assaulted':18,'tingling':19,
           'losing weight':20,'suspected stroke':21,'sudden slurred speech':21,'slurred speech':21,'circulation decreased':22,
           'vaginal bleeding':23,'spotting':23,'sleeping more':24,'hurts to pee':25,'urinate':25,'drooling':26,
           'difficulty swallowing':27,'barely swallow':27,'cold':28,'chills':28,'migraine':29,'food poisoning':30,'cramping':31,
           'crampy':32,'voice is different':33,'dizziness':34,'pneumonia':35,'infection':36,"ran out of adhd medication":37,
           'ran out of my blood 4 medicine':37,'ran out of blood 4 medication':37,'certificate':39,
           'pill for pain':37, 'prescription for pain medication':37,'rabies':50, 'physical exam':39,
           'need tetanus shot':49,'need tetanus booster':49,'need a tetanus shot':49,'runny nose':38,'needs a work note':39,'control visit':39,'pill prevent pregnancy':51,'need  tetanus booster':49,
           'needs to go into detox again':39,'sawdust':40,'bug inside':40,'thing stuck':40,'something':40,'poison ivy':41,'sudden outbursts':42,
           'change in mental status':43,'sneez':44,'hypoglycemia,':45,'good circulation':46,'big':47,'sensation':48
          }

#data.loc[142,'place ']='lungs'
#data.loc[198,'place ']='foot'
#data.loc[171,'type of injury']='maybe bitten'
def tag_main_symptoms(data_cop,sym_words):

    #data_cop.loc[123,'main symptoms'] = np.NaN
    match_words(data_cop,'main symptoms',sym_words)
    word_remove_list={'fever':'','ing up thick green phlegm  xxx':'','ing':'','on one side of my':'','decreased':'','when':''}
    data_cop.loc[113,'main symptoms']=str(data_cop.loc[113,'main symptoms']).replace('fever','')


    match_words(data_cop,'main symptoms',word_remove_list)
    add_remove_words(data_cop,'main symptoms','skin', ['red'])
    data_cop['main symptoms'] = data_cop['main symptoms'].str.replace('pani','pain')
    data_cop['main symptoms'] = data_cop['main symptoms'].str.replace('.','').str.replace(',','')
aw_words = {np.NaN:0,'unresponsive':1,'unrestrained':1,'somnolent':2,'lethargic':2,'listless':2,'weak':13,'weaknes':13,' tiring out':13,
            'not acting right':3,'flaccid':3,'no response to pain':1,
            'gcs 3':1,'unequal pupils':4,'moaning':5,'does not respond to voice':6,'unable to answer':6,'30 %  body surface area':7, 'alert':8,'14 gcs':8,
            'oriented':9,'pass out':17,'moving all extremities':10,'sititng':10,'sitting,':10,'conscious':10,'anxious':11,'sudden outbursts':11,
            'un10 for a couple of minutes':17,'unsteady gait':10,'powerless':13, 'before i 17':13,
            'unable to remember the events':12,'1 min loss of consciousness':17,'1 min loss of 10ness':17,'crying':5,
            'responding appropriately to questions':9,'able to answer your questions':9,'asking':16,
            'responds to verbal stimuli':19,'equal hand grasps':19,'confused':14,'awake':16,'currently aware':16,
            'dis9 to time and place':12,'quiet':15,'cooperative':16,'pleasant':15,'active':16,'looks well':16,
            'running':16,'looking around':10,'remembers the fall':20,'acting appropriately':16,
            'answers your questions appropriately':20, '9 to person and place and time':20,'exhausted':21,
            'sensation decreased':21,'feeling run down':21,'dizzy':22,'despairing':23}

def tag_awareness(data_cop,aw_words):

    match_words(data_cop,'awareness',aw_words)
    data_cop['awareness'] = data_cop['awareness'].str.replace('sudden','').str.replace('13nes','13').str.replace('to painful stimuli','').str.replace('questions','')
    data_cop['awareness'] = data_cop['awareness'].str.replace(',','')
types_words ={np.nan:0,'hit':1,'punched':1,'fell':2,'fall':2,'slipped':2,'lacerations':3,'laceration':3,
    'gunshot':4,'shot':4,'scratched':5,'stab wound':6,'pierce':6, 'stepped on a rusty nail':6,
    'stabbed':6,'cut off':7,'open area':8,'open fracture':9,'burn':10,'broken':20,'abrasion':11,'ragged':11,'injured':11,
    'cut':12,'twisted':13,'maybe bitten':14,'refusing to move':15,'stiff':15,"can't move":15,'flaccid':15,'movement':16,'not 20':17,
    'no obvious deformity':17,'shortened':18,'obvious deformity':18,'obviously deformed':18,'obvious deformity':18,'dislocation':18,'everted':19}


#data.loc[29,'awareness']='flaccid'
#data.loc[29,'type of injury']=np.NaN
#remove_words={'superficial':'','single':'','self-inflicted':'','multiple':''}
#add_remove_words('type of injury','place ',['right hip','right leg','face','left arm'])
def tag_typs_injury(data_cop,types_words):
    for index, typ in enumerate(data_cop['type of injury']): 
        replace_all(data_cop,'type of injury',index,types_words) 
    match_words(data_cop,'type of injury',{'superficial':'','single':'','self-inflicted':'','multiple':'','tree':'',
                                           'externally rotated':'','on the ice':'','11s':'11'})
#tag_typs_injury()
    data_cop['type of injury'] = data_cop['type of injury'].str.replace(',','')
def tag_fever(data_cop):
    data_cop['fever'] = data_cop['fever'].map({np.nan:0,'yes':1,'101.66':1,'103':1,'no':0})

deh_words ={np.NaN:0,'vomiting':1,'vomitig':1,'diarrhea':2,'no pee':3,'nauseous':4}
def tag_dehydration(data_cop,deh_words):
    match_words(data_cop,'dehydration', deh_words)
def tag_appetite(data_cop):
    appetite_words ={np.NaN:0,'no':1,'yes':2}
    match_words(data_cop,'appetite', appetite_words)
def tag_defibrillated(data_cop):
    data_cop['defibrillated ']=data_cop['defibrillated '].replace('no',0).replace(np.nan,0)
def add_val_awareness_4_5(data_cop):
    for x,y in enumerate(data_cop['class']):
        if y==5 and str(data_cop.loc[x,'awareness'])=='nan':
            data_cop.loc[x,'awareness']='alert'
        if y==4 and str(data_cop.loc[x,'awareness'])=='nan':
            data_cop.loc[x,'awareness']='alert'
#add_val_awareness_4_5()
def add_no_if_nan(data_cop):
    name_no = ['CPR','pregnat','defibrillated ','suicide attempt','fever','bleeding','alcohol consumption']
    name_lack_inf = ['medical history','main symptoms','breath','appetite']
    for name in name_no:
        data_cop[name].replace(np.nan, 'no',inplace=True)
    for x in name_lack_inf:
        data_cop[x].replace(np.nan, 0 ,inplace=True)

def tag_pregnat(data_cop):
    data_cop['pregnat'] = data_cop['pregnat'].replace('no',0,).replace('suspicion',2).replace('yes',1)
def drop_cm( data_cop,index,tex,val_int):
    data_cop.loc[index,'type of injury'] = data_cop.loc[index,'type of injury'].replace(tex,'')
    data_cop.loc[index,'type of injury'] = data_cop.loc[index,'type of injury'].replace(val_int,'')

def add_col_typ_size(data_cop):
    lista = []

    for x,y in enumerate(data_cop['type of injury']):
        zero = 1
        if '2-' in str(y):
            drop_cm(data_cop,x,'centimeter','2-')
            lista.append(2)
            zero =0
        if 'cm' in str(y):
            drop_cm(data_cop,x,'cm','2')
            lista.append(2)
            zero =0
        if '3-' in str(y):
            drop_cm(data_cop,x,'centimeter','3-')
            lista.append(3)
            zero =0
        if '6-' in str(y):
            drop_cm(data_cop,x,'centimeter','6-')
            lista.append(6)
            zero =0
        if '4-' in str(y):
            drop_cm(data_cop,x,'centimeter','4-')
            lista.append(4)
            zero =0
        if 'large' in str(y):
            drop_cm(data_cop,x,'large','large')
            lista.append(4)
            zero =0
        if zero==1:
            lista.append(0)
    data_cop.insert(13, "Typ_size", lista, True) 
def drop_fall( data_cop,index,tex):
    data_cop.loc[index,'type of injury'] = data_cop.loc[index,'type of injury'].replace(tex,'')   
def add_col_typ_fall_height(data_cop):
    lista = []
    for x,y in enumerate(data_cop['type of injury']):
        zero = 1
        if '8 feet' in str(y):
            drop_fall(data_cop,x,'8 feet')
            lista.append(8)
            zero =0
        if '5 feet' in str(y):
            drop_fall(data_cop,x,'5 feet')
            lista.append(5)
            zero =0
        if '4 feet' in str(y):
            drop_fall(data_cop,x,'4 feet')
            lista.append(4)
            zero =0
        if '5 m' in str(y):
            drop_fall(data_cop,x,'5 m')
            lista.append(16)
            zero =0
        if zero==1:
            lista.append(0)
    data_cop.insert(14, "Typ_fall_height", lista, True) 
def separate_col_BP(data_cop):  
    lista_BP_1=[]
    lista_BP_2=[]
    for x,y in enumerate(data_cop['BP']):
        if str(y)=='nan':
            lista_BP_1.append(np.nan)
        else:
            lista_BP_1.append(str(y).split('/')[0])
        if str(y)=='nan':
            lista_BP_2.append(np.nan)
        else:
            lista_BP_2.append(str(y).split('/')[1])
    return   lista_BP_1,  lista_BP_2

def drop_add_BP(data_cop,BP_1,BP_2):  
    data_cop.drop('BP',axis='columns', inplace=True)
    data_cop.insert(6, "BP_1", BP_1, True)
    #print(BP_1,BP_2)
    data_cop.insert(7, "BP_2", BP_2, True) 
#separate_col_BP(data)
def clean_col_stop_words(data_cop):
    name_col = ['skin','medical history','main symptoms', 'type of injury', 'additional information','awareness','breath']

    for x in name_col:
        data_cop[x] = data_cop[x].str.lower().str.strip()

        stop_words = ["and","the", "is","his", "from","to","are","my","a",
                      "was",'xxx',"did","for","of","me","on","into","i","nan",
                     "been","has","be","in","does","your","xxxx",'-','drop_me','ice','cat','by','tree',
                     'touch',' oozing', 'around', 'site','sandpaper','very',',']

        pat = '|'.join(r"\b{}\b".format(x) for x in stop_words)
        data_cop[x] = data_cop[x].str.replace(pat, '')
        data_cop[x] = data_cop[x].str.split()

    data_cop['dehydration'].str.lower().str.strip()
    data_cop['dehydration'] = data_cop['dehydration'].str.split(' xxx ')
#clean_col_stop_words()
def uniform_value_pregnat(data_cop): 
    for x, y in enumerate(data_cop['pregnat'].str.contains('was',na=False, regex=True)):
        if y:
            was_pregnat = data_cop['pregnat'].loc[x].replace('was ','-')
            data_cop['pregnat'].loc[x]= was_pregnat
            # print(data['pregnat'].loc[x])
            if 'w' in data_cop['pregnat'].loc[x]:
                value_pregnat = float(data_cop['pregnat'].loc[x].replace('w',''))
                value_pregnat = value_pregnat*7
                data_cop['pregnat'].loc[x] = value_pregnat
            elif 'm' in data_cop['pregnat'].loc[x]:
                value_pregnat = float(data_cop['pregnat'].loc[x].replace('m',''))
                value_pregnat = value_pregnat*30
                data_cop['pregnat'].loc[x] = value_pregnat
            elif 'd' in data_cop['pregnat'].loc[x]:
                value_pregnat = float(data_cop['pregnat'].loc[x].replace('d',''))
                data_cop['pregnat'].loc[x] = value_pregnat

    for x, y in enumerate(data_cop['pregnat'].str.contains('w',na=False, regex=True)):
        if y:
            value_pregnat = float(data_cop['pregnat'].loc[x].replace('w',''))
            value_pregnat = value_pregnat*7
            data_cop['pregnat'].loc[x] = value_pregnat
    for x, y in enumerate(data_cop['pregnat'].str.contains('m',na=False, regex=True)):
        if y:
            value_pregnat = float(data_cop['pregnat'].loc[x].replace('m',''))
            value_pregnat = value_pregnat*30
            data_cop['pregnat'].loc[x] = value_pregnat
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def uniform_value_old(data_cop):   

    data_cop['old'] = data_cop['old'].str.replace('>1m and <=12 m','150 d')

    for x,y in enumerate(data_cop['old']):
        if 'nan' != str(data_cop['old'].loc[x]) and is_number(y):
                #print(x,y)
                val = float(y)
                val= val*365
                data_cop['old'].loc[x] = val
                #print( data['old'].loc[x])
        else:
                if '>'in str(data_cop['old'].loc[x]).strip():
                    data_cop['old'].loc[x] = data_cop['old'].loc[x].replace('>18','7300 d')
                if '<'in str(data_cop['old'].loc[x]):
                    data_cop['old'].loc[x] = data_cop['old'].loc[x].replace('<18','3285 d')
                if 'd'in str(data_cop['old'].loc[x]):
                    data_cop['old'].loc[x] = float(data_cop['old'].loc[x].replace('d',''))
                if 'w'in str(data_cop['old'].loc[x]):
                    value_old = float(data_cop['old'].loc[x].replace('w',''))
                    value_old = value_old*7
                    data_cop['old'].loc[x] = value_old
                if 'm'in str(data_cop['old'].loc[x]):
                    value_old = float(data_cop['old'].loc[x].replace('m',''))
                    value_old = value_old*30
                    data_cop['old'].loc[x] = value_old
    old_int = float(data_cop['old'].mean())
    data_cop['old'] = data_cop['old'].fillna(old_int)
#uniform_value_old():

def vital_signal(data_cop,x,b,C):
    y = data_cop.loc[x,'old']
    if y<=180 and b :
        for c,d in C.items():
            if c is'RR' and d is True:
                data_cop.loc[x,'RR'] = 45
            if c=='SPO2' and d:
                data_cop.loc[x,'SPO2 ']=98
            if c=='BP' and d:
                data_cop.loc[x,'BP']='80/55'
            if c=='HR' and d:
                data_cop.loc[x,'HR']=140
            if c=='T' and d:
                data_cop.loc[x,'T']=98.6
    if y>180 and y<=360 and b:
        for c,d in C.items():
            if c=='RR' and d:
                data_cop.loc[x,'RR'] = 25
            if c=='SPO2' and d:
                data_cop.loc[x,'SPO2 ']=99
            if c=='BP' and d:
                data_cop.loc[x,'BP']='92/60'
            if c=='HR'and d:
                data_cop.loc[x,'HR']=80
            if c=='T' and d:
                data_cop.loc[x,'T']=98.6
    if y>360 and y<=1800 and b:
        for c,d in C.items():
            if c=='RR' and d:
                data_cop.loc[x,'RR'] = 22
            if c=='SPO2' and d:
                data_cop.loc[x,'SPO2 ']=100
            if c=='BP' and d:
                data_cop.loc[x,'BP']='103/70'
            if c=='HR' and d: 
                data_cop.loc[x,'HR']=65
            if c=='T' and d:
                data_cop.loc[x,'T']=98.6
    if y>1800 and y<=3960 and b:
        for c,d in C.items():
            if c=='RR' and d:
                data_cop.loc[x,'RR'] = 13
            if c=='SPO2' and d:
                data_cop.loc[x,'SPO2 ']=98
            if c=='BP' and d:
                data_cop.loc[x,'BP']='91/60'
            if c=='HR' and d: 
                data_cop.loc[x,'HR']=110
            if c=='T' and d:
                data_cop.loc[x,'T']=97.9
    if y>3960 and y<=6840 and b:
        for c,d in C.items():
            if c=='RR' and d:
                data_cop.loc[x,'RR'] = 16
            if c=='SPO2' and d:
                data_cop.loc[x,'SPO2 ']=99
            if c=='BP' and d:
                data_cop.loc[x,'BP']='97/71'
            if c=='HR' and d: 
                data_cop.loc[x,'HR']=66
            if c=='T' and d:
                data_cop.loc[x,'T']=99.1
    if y>6840 and b:
        for c,d in C.items():
            if c=='RR' and d:
                data_cop.loc[x,'RR'] = 16
            if c=='SPO2' and d:
                data_cop.loc[x,'SPO2 ']=98
            if c=='BP' and d:
                data_cop.loc[x,'BP']='120/70'
            if c=='HR' and d : 
                data_cop.loc[x,'HR']=70
            if c=='T' and d:
                data_cop.loc[x,'T']=98.2
def add_value_vital_signal_and_drop_col(data_cop):

    data_cop['vital signal']=data_cop['vital signal'].map({'normal limits':1,'stable':1,'normal limits.':1})
    is_stable = data_cop['vital signal'] == 1

    for x,b in enumerate(is_stable):
        c={'RR':True,'SPO2':True,'BP':True,'HR':True,'T':True}
        vital_signal(data_cop,x,b,c)   
    data_cop.drop('vital signal', 1, inplace = True)
    for a,b in  enumerate(data_cop['RR']):
        if data_cop.loc[a,'class']==5 or data_cop.loc[a,'class']==4 or data_cop.loc[a,'class']==3:
            if str(b) =='nan' and str(data_cop.loc[a,'RR'])=='nan' and str(data_cop.loc[a,'HR'])=='nan' and str(data_cop.loc[a,'BP'])=='nan':
                c={'RR':True,'SPO2':True,'BP':True,'HR':True,'T':True}
                b=1
                vital_signal(data_cop,a,b,c)
            if str(b) =='nan':
                b=1
                c={'RR':True, 'SPO2':False, 'BP':False,'HR':False,'T':False}
                vital_signal(data_cop,a,b,c)
            if str(data_cop.loc[a,'HR'])=='nan':
                b=1
                c={'RR':False,'SPO2':False,'BP':False,'HR':True,'T':False}
                vital_signal(data_cop,a,b,c)
            if str(data_cop.loc[a,'SPO2 '])=='nan':
                b=1
                c={'RR':False, 'SPO2':True, 'BP':False,'HR':False,'T':False}
                vital_signal(data_cop,a,b,c)
            if str(data_cop.loc[a,'BP']):
                b=1
                c={'RR':False,'SPO2':False,'BP':True,'HR':False,'T':False}
                vital_signal(data_cop,a,b,c)
            if str(data_cop.loc[a,'T']):
                b=1
                c={'RR':False,'SPO2':False,'BP':False,'HR':False,'T':True}
                vital_signal(data_cop,a,b,c)
    #print(data_cop['BP'])

#add_value_vital_signal_and_drop_col()  
# print(data_cop['RR'].notnull().sum(),data_cop['BP'].notnull().sum(), data_cop['HR'].notnull().sum(), data_cop['SPO2 '].notnull().sum())
def tag_intubation_EMS_bleeding_alcohol_con_CPR_sex(data_cop):   
    name = ['intubation','EMS','bleeding','alcohol consumption','CPR','sex']
    for x in name:
        data_cop[x] = data_cop[x].map({'no':0,'on':0,'yes':1,'Yes':1,'male':0,'female':1})

    # data['sex'] = data['sex'].replace(to_replace = np.nan, value = 2)  
    pd.to_numeric(data_cop[x])
    data_cop['pain'] = data_cop['pain'].replace(to_replace = np.nan, value = 0) 
    data_cop['when'] = data_cop['when'].replace(to_replace = np.nan, value = 0) 

    data_cop['sex'] = data_cop['sex']. fillna (method = 'ffill')
def change_val_SPO2(data_cop):
    for x,y in enumerate(data_cop['SPO2 ']):
        if y ==" < 90":
            data_cop.loc[x,'SPO2 '] = 89
    data_cop['SPO2 '] = pd.to_numeric(data_cop['SPO2 '])
def uniform_value_when(data_cop):  
    for x,y in enumerate(data_cop['when']):
        if str(y).strip() == "few hours ago":
            data_cop.loc[x,'when']=5*60
        if str(y).strip() == "couple of days ago":
            data_cop.loc[x,'when']=3*24*60
        if str(y).strip() == "early this morning":
            data_cop.loc[x,'when']=4*60
        if str(y).strip() == "few days":
            data_cop.loc[x,'when']=5*24*60
        if str(y).strip() == "this morning":
            data_cop.loc[x,'when']=6*60
        if str(y).strip() == "last night":
            data_cop.loc[x,'when']=12*60
        if str(y).strip() == "today":
            data_cop.loc[x,'when']=3*60
        if str(y).strip() == "1 months":
            data_cop.loc[x,'when']=30*24*60
        if str(y).strip() == "one month":
            data_cop.loc[x,'when']=30*24*60
        if ' h' in str(data_cop.loc[x,'when']):
            value_pregnat = float(data_cop.loc[x,'when'].replace('h',''))
            data_cop['when'].loc[x] = value_pregnat *60
        if 'min' in str(data_cop.loc[x,'when']).strip():
            data_cop['when'].loc[x] = float(data_cop.loc[x,'when'].replace('min',''))
        if 'days' in str(data_cop.loc[x,'when']).strip():
            value_pregnat = float(data_cop.loc[x,'when'].replace('days',''))
            data_cop['when'].loc[x] = value_pregnat *24*60
    data_cop['when'] = pd.to_numeric(data_cop['when'])


# In[7]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn import model_selection
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error
from math import sqrt
get_ipython().run_line_magic('matplotlib', 'inline')

#wywołnie funkcji

def clean_dataFrame(data_cop):
    clean_vital_signal(data_cop)
    replace_palp(data_cop)
    drop_add_col(data_cop,16,'additional information')
    add_words_forget(data_cop)
    change_words(data_cop)

    drop_col_class_two(data_cop)
    #print(data_cop.head())
    tag_main_symptoms(data_cop,sym_words)
    tag_pain(data_cop)
    tag_breath(data_cop, breath_words)
    
    tag_fever(data_cop)
    tag_skin(data_cop,skin_word)
    tag_typs_injury(data_cop,types_words)
    tag_dehydration(data_cop,deh_words)
    tag_appetite(data_cop)

    tag_his_medical(data_cop,med_his_words)
    tag_additional_inf(data_cop,add_inf_words)
    add_val_awareness_4_5(data_cop)
    add_no_if_nan(data_cop)
    tag_pregnat(data_cop)
    tag_awareness(data_cop,aw_words)

    tag_place(data_cop,body)
    clean_place(data_cop)

    add_col_typ_size(data_cop)
    add_col_typ_fall_height(data_cop)


    clean_col_stop_words(data_cop)
    tag_suicide_at(data_cop)
    data_cop.loc[219,'suicide attempt']=0
    uniform_value_pregnat(data_cop)
    uniform_value_old(data_cop)
    add_value_vital_signal_and_drop_col(data_cop)  
    tag_intubation_EMS_bleeding_alcohol_con_CPR_sex(data_cop)
    change_val_SPO2(data_cop)
    uniform_value_when(data_cop)
    tag_defibrillated(data_cop)
    BP_1,BP_2 = separate_col_BP(data_cop)
    drop_add_BP(data_cop,BP_1,BP_2)
    for x in ['BP_1','BP_2','RR','HR','T']:
        data_cop[x] = pd.to_numeric(data_cop[x])
    for i in ['HR','RR','SPO2 ','BP_1','BP_2']:
        interquartile (data_cop,i)
    
    uzupelnij_HR_regresja(data_cop,'HR','RR')  
    interquartile (data_cop ,'RR')
    #uzupelnij_HR_regresja(data_cop,'RR','HR')
    interquartile (data_cop ,'SPO2 ')
    #uzupelnij_HR_regresja(data_cop,'RR','SPO2 ')
    
    add_vital_signal_no_breath(data_cop)
    add_median_vital_signal(data_cop)
    #add_median_vital_signal_global(data_cop)
    data_cop['appetite'] = pd.to_numeric(data_cop['appetite'])
    data_cop['pregnat'] = pd.to_numeric(data_cop['pregnat'])
    data_cop['suicide attempt'] = pd.to_numeric(data_cop['suicide attempt'])
    data_cop['Typ_size'] = pd.to_numeric(data_cop['Typ_size'])


    return data_cop


def uzupelnij_HR_regresja(data_cop,X,Y):
    regr = reg_vital_signal(data_cop,[X],[Y])
    for x,y in enumerate(data_cop[Y]):
        if 'nan' in str(y) and str(data_cop.loc[x,X])!=str(np.nan):
            #print('indeks: ',x,', wartość',Y,':',y,',wartość',X,':',data_cop.loc[x,X])
            data_cop.loc[x,Y]= match_value(regr, [float(data_cop.loc[x,X])])
            
def reg_vital_signal(data_cop,cechy,cecha):
    par = clean_data_to_regresion(data_cop,cechy[0],cecha[0])
    X=par[cechy].values.reshape(-1,1)
    
    y=par[cecha]
    splits = model_selection.train_test_split(X, y, test_size=0.20, random_state=0)
    X_train, X_test, Y_train, Y_test = splits
    regr = linear_model.LinearRegression()
    # Train the model using the training sets
    regr.fit(X_train, Y_train)
    # Predict values
    Y_predicted = regr.predict(X_test)
    # The coefficients
    #print('Coefficients: \n', regr.coef_)
    #intercept_ 
    #print('intercept_: \n', regr.intercept_)   
    #print ( 'Regresja liniowa R do kwadratu: % .4f '  %  regr . score ( X_test ,  Y_test ))
    rmse = sqrt(mean_squared_error(Y_test, Y_predicted))
    #print('X_test',len(X_test), 'Y_test: ',len(Y_test))
    #print('RMSE:', rmse )
    #plt.scatter(X_test, Y_test,  color='black')
    #plt.plot(X_test, Y_predicted, color='blue', linewidth=3)
    
    return regr


def clean_data_to_regresion(data_cop,par_1,par_2):
    data_vital = data_cop[['RR', 'SPO2 ', 'HR']]
    drop_RR_HR = data_vital.dropna(subset = [par_1,par_2])
    #print(len(drop_RR_HR),par_1,par_2)
    #drop_RR_HR = data_vital[['RR','HR','SPO2 ','old','class','BP_1','BP_2']]
    drop_RR_HR.reset_index(drop=True, inplace=True)
    return drop_RR_HR

def match_value(regr, values):
    features = values
    x_pred = np.array(values)
    x_pred = x_pred.reshape(-1, len(features))
    #print(regr.predict(x_pred))
    
    return int(regr.predict(x_pred))

def interquartile ( data_cop ,  col):   
    for nr_class in range(1,6):
        Q1 = data_cop[col].where(data_cop['class']==nr_class).quantile(0.25)
        Q3 = data_cop[col].where(data_cop['class']==nr_class).quantile(0.75)
        Q2 = data_cop[col].where(data_cop['class']==nr_class).median()
        IQR = Q3 - Q1
        low = Q1 - 1.5 * IQR
        high = Q3 + 1.5 * IQR
        index_low = data_cop . loc [( data_cop [col ].where(data_cop['class']==nr_class)  <  low ) ].index
        index_high = data_cop . loc [( data_cop [col ].where(data_cop['class']==nr_class)  >  high )].index
        #print('wartość Q1 dla cechy: ',col,'oraz klasy: ',nr_class,'wynosi:', Q1)
        #print('próbki odstające(poniżej poziomu min):')
        for x in index_low:
            #print(x,clean_data[col].loc[x])
            data_cop[col].loc[x] = Q1
        #print('wartość Q3 dla cechy: ',col,'oraz klasy: ',nr_class,'wynosi:', Q3)
        #print('próbki odstające(powyżej poziomu max):')
        for x in index_high:
            #print(x,clean_data[col].loc[x])
            data_cop[col].loc[x] = Q3
           

            

def clean_vital_signal(data_cop):
    INDEX_vital =['BP', 'HR', 'RR', 'T']
    for x,y in enumerate(data_cop['vital signal']):
        if '78' in str(y):
            data_cop.loc[x,'vital signal'] = np.NaN
            a =str(y).split('xxx')
            #print(x)
            for index, val in enumerate(INDEX_vital):
                data_cop.loc[x,val] = a[index].replace(val,'',True)
                if 'F' in str(y):
                    data_cop.loc[x,val] = data_cop.loc[x,val].replace('F','',True)


def list_median_vital(data_cop):
    list_val_med=[]
    for x in range(1,6):
        a,b,c,d,e,f =data_cop[['RR','HR','BP_1','BP_2','T','SPO2 ']].where(data_cop['class']==x).median()
        per=a,b,c,d,e,f
        list_val_med.append(per)
    return list_val_med
def list_median_vital_global(data_cop):
    list_val_med=[]
    a,b,c,d,e,f =data_cop[['RR','HR','BP_1','BP_2','T','SPO2 ']].median()
    per=a,b,c,d,e,f
    list_val_med.append(per)
    return list_val_med

def add_vital_signal_no_breath(data_cop):
    for x,y in enumerate(data_cop['breath']):
        if '3' in str(data_cop.loc[x,'breath']):
            data_cop.loc[x,'RR']=0
            data_cop.loc[x,'HR']=30
            data_cop.loc[x,'BP_1']=72
            data_cop.loc[x,'BP_2']= 53
            data_cop.loc[x,'SPO2 ']= 87
            data_cop.loc[x,'T']=98.0 
        if '10' in str(data_cop.loc[x,'breath']):
             data_cop.loc[x,'RR']=50
        if str(data_copi.loc[x,'defibrillated ']) != 'nan':
            data_cop.loc[x,'RR']=0
            data_cop.loc[x,'HR']=30
            data_cop.loc[x,'BP_1']=72
            data_cop.loc[x,'BP_2'] =53
            data_cop.loc[x,'SPO2 ']= 87
            data_cop.loc[x,'T']=98.0 

skin_word ={np.NaN:0,'rash':1,'dry':2,'cracked':3,'swelling':4,'moist lesions':5,'tender':6,'warm':7,'cool':8,
            'cold':8,'diaphoretic':9, 'purpuric lesions':10,'petechial':11,'little pinpoint purplish':11,'mottle':12,'red':13,
            'blood':13,'swollen':14,'pus':15,'pale':16,'clammy':17,'small paronychia':18,'tearing':19,'itchy':20,
            'very mucous membranes':21,'bruising':22,'crusty':23,'unable to palpate a pulse':24,'flushed':25}

def tag_skin(data_cop,skin_word):
    match_words(data_cop,'skin',skin_word)

    data_cop['skin'] = data_cop['skin'].str.replace('12d','12')
    words={'dened':'','ness':'','new':'','area':'','across':'','to touch':'','sandpaper':'','very':'','is oozing from around the site':''}
    match_words(data_cop,'skin',words)
    data_cop['skin'] = data_cop['skin'].str.replace(',','')

def replace_palp(data_cop):
    data_cop['BP']= data_cop['BP'].str.replace('palp','60',True)
    data_cop['BP']= data_cop['BP'].str.replace('pal','60',True)

def tag_suicide_at(data_cop):
    data_cop['suicide attempt'] = data_cop['suicide attempt'].map({'no':0,'drug overdose':1,'selfinflicted':2})

def drop_add_col(data_cop,index_col, name_where_add):
    column_names = data_cop.columns.tolist()
    for index, val_activ in enumerate(data_cop[column_names[index_col]]):
        if data_cop.loc[index, column_names[index_col]] == 'yes':
            data_cop.loc[index, name_where_add] = 'physical activity'

    data_cop.drop(column_names[index_col],1, inplace =True)

        
def add_words_forget(data_cop):
    data_cop.loc[10,'place ']= 'vagina'
    data_cop.loc[10,'skin'] = 'pale'
    data_cop.loc[10,'awareness'] = 'before I pass out'
    data_cop.loc[10,'main symptoms']='passing clots the size of oranges'
    data_cop.loc[107,'awareness'] = 'anxious xxx despairing'
    data_cop.loc[185,'pregnat'] = 'no'
    data_cop.loc[188,'pregnat'] = 'no'
    
    data_cop.loc[41,'additional information']='requiring insulin occlusion'
    data_cop.loc[41,'skin'] =data_cop.loc[41,'skin'] + 'unable to palpate a pulse'
    data_cop.loc[142,'place ']='lungs'
    data_cop.loc[198,'place ']='foot'
    data_cop.loc[171,'type of injury']='maybe bitten'


def replace_all(data_cop,col,index,d):
    for a,b in d.items():
        data_cop.loc[index,col] = str(data_cop.loc[index,col]).lower().replace(str(a),str(b))
        #print(index,data_cop.loc[index,col])


def match_words(data_cop,text, words):
    for x,txt in enumerate(data_cop[text]):
        replace_all(data_cop,text, x, words)
        if '' == str(data_cop.loc[x,text]).strip():
            #print(index,data_cop.loc[x,txt])
            data_cop.loc[x,text]=np.NaN

def add_remove_words(data_cop,col_remove, col_add, words):
    for word in words:
        for x,y in enumerate(data_cop[col_remove]):
            if word in str(y):
                if str(word) == str(y).strip():
                    data_cop.loc[x,col_remove]= np.NaN
                else:
                    #print(x,y)
                    data_cop.loc[x,col_remove]= data_cop.loc[x,col_remove].replace(word,'')
                    if 'xxx' == str(data_cop.loc[x,col_remove]).strip() or 'and'==str(data_cop.loc[x,col_remove]).strip():
                        data_cop.loc[x,col_remove]= np.NaN
                if 'nan' in str(data_cop.loc[x,col_add]):
                    if str(col_add)=='pain':
                        if 'severe' in str(word) or 'hard' in str(word):
                            data_cop.loc[x,col_add]= 8
                        else:
                            data_cop.loc[x,col_add]= 6
                    else:
                        data_cop.loc[x,col_add]= word
                else:
                    if str(col_add)!='pain':
                         data_cop.loc[x,col_add]= data_cop.loc[x,col_add] + " "+ str(word)
                        
def drop_col_class_two(data_cop):
    count_col_not_null = {x:data_cop.loc[x].count() for x,y in enumerate(data_cop['class']) if '2'==str(y)}
    sort_orders = sorted(count_col_not_null.items(), key=lambda x: x[1], reverse=False)
    index_col_to_drop=[x for x,y in sort_orders[:20] ]
    #print(sort_orders[:20])
    data_cop.drop(index_col_to_drop, inplace=True)
    data_cop.reset_index(drop=True, inplace=True)

def change_words(data_cop):
    skin_words=['new reddened area','tenderness','itchy','crusty','tearing','reddened','redness','swollen','swelling',
                'small paronychia','bruising']
    type_words=['3-centimeter laceration','broken','right leg is shortened and externally rotated','no obvious deformity',
               'obviously deformed','obvious deformity','movement',"can't move",'ragged','stepped on a rusty nail',
               'injured','stiff','abrasion','refusing to move','twisted','dislocation','flaccid']
    awareness_words=['sensation decreased','exhausted','feeling run down','dizzy']
    place_words=['throat','tooth',' abdominal']
    add_inf_words=['suspicion positive streptococci','colitis is acting up','medications not work',
                   'large hematoma around the wound',"EKG - anterior lateral ischemic changes",
                  'passing clots the size of oranges','irregular HR','obvious chronic obstructive pulmonary disease']

    add_remove_words(data_cop,'main symptoms','type of injury',type_words)
    add_remove_words(data_cop,'main symptoms','skin',skin_words)
    add_remove_words(data_cop,'main symptoms','awareness',awareness_words)
    add_remove_words(data_cop,'main symptoms','place ',place_words)
    add_remove_words(data_cop,'place ','type of injury',['2-centimeter laceration'])
    add_remove_words(data_cop,'main symptoms','additional information',add_inf_words)
    add_remove_words(data_cop,'awareness','main symptoms',['sudden slurred speech','slurred speech'])
    add_remove_words(data_cop,'type of injury','awareness',['flaccid'])
    add_remove_words(data_cop,'type of injury','place ',['right hip','right leg','face','left arm'])
    add_remove_words(data_cop,'skin','place ', ['chest','lips','mucous membranes','left arm','face','right hip"'])
    add_remove_words(data_cop,'type of injury','main symptoms',['something in'])

def tag_pain(data_cop):
    sym_nan_words={np.NaN:0}
    pain =['hurts','hurt','hard','hart','ache','sore','painful','sore','severe pain','severe','pain','xxx   xxx']
    add_remove_words(data_cop,'main symptoms','pain',pain)
    data_cop['pain']=data_cop['pain'].replace(20,10)
    match_words(data_cop,'main symptoms',sym_nan_words)

def tem_F(data_cop):
    for x,y in enumerate(data_cop['T']):
        if 'C' in str(y):
            a =float(data_cop.loc[x,'T'].replace('C',''))
            data_cop.loc[x,'T'] =(a*1.8)+32
med_his_words={np.NaN:0,'htn':1,'medication for blood pressure':1,'asthma':2,'asthhma':2,'2tic':2,'uses several metered-dose inhalers':2,
               'migraine':3,'migreny':3,'3s':3,'diabetic':4,
               'diabetes':4,'type 2 4':4,'insulin':4,'6 weeks post laparoscopic gastric bypass':5,
              'metastatic ovarian cancer':6,'poison ivy':7,'arthritis':8,'chronic renal failure':9,'regular kidney dialysis':9,
              'aphasic':10, 'massive 11 3 years ago':11,'massive 11':11,'stroke':11,'suffered a massive 11':11,
               'therapeutic abortion':13,'cardiac':14,'14s':14,'heart attack':14,'infarction':14,'atrial fibrillations':14,'atrial fibrillation':14,
               'multiple medications':15,'medications':15,'gastroesophageal reflux disease':16,'medications include fioricet':17,  'include fioricet':17,
               'allergic to penicillin':18,'severe shellfish allergy':18, 'warfarin':19,
               'amiodarone':20,'chronic obstructive pulmonary disease':21,'healthy':22,'psoriasis':23,
               'colitis':24,'intubations':25,'intubated':25,'clots in my legs':26, 'high cholesterol':27,
               'epipen':2,'lymphoma':28,'lung cancer':28,'metastatic breast cancer':28,'completed chemotherapy': 35,
               'multiple sclerosis':29,'chemotherapy':30,'sickle cell disease':31,'no tetanus immunization':32,
               'last tetanus immunization was 10 years ago':32,'aspirin':33,'ractured':34,'kidney stones':35,
               'previous ectopic preg0cy':39,
               'antibiotics for 5 days for mastitis':40,'ibuprofen':41,'frequent ear infections':42,'ear infections':42,'no allergies':43,
                'no meds':44,'no medications':44,'immunizations are up to date':45,'no 15':44}

def tag_his_medical(data_cop,med_his_words):


    for x,typ in enumerate(data_cop['medical history']):
        is_list =0
        if 'diuretic' in str(typ):
            data_cop.loc[x,'medical history'] = data_cop.loc[x,'medical history'].replace('diuretic','12 15')
        if 'knee replacement' in str(typ):
            data_cop.loc[x,'medical history'] = data_cop.loc[x,'medical history'].replace('knee replacement','36 37')
        if 'gastric bypass' in str(typ):
            data_cop.loc[x,'medical history'] = data_cop.loc[x,'medical history'].replace('gastric bypass','36 38')
        if  'pill to thin his blood' in str(typ):
            data_cop.loc[x,'medical history'] = data_cop.loc[x,'medical history'].replace( 'pill to thin his blood','15 26')
        if 1:
            replace_all(data_cop,'medical history',x, med_his_words)



    match_words(data_cop,'medical history',{'2 weeks ago':'','late-stage non-hodgkinős':'','wrist':'','3 months ago':'','3 weeks ago':'','3 years ago':'','6 weeks post laparoscopic':''})
    data_cop['medical history'] = data_cop['medical history'].str.replace(',','').str.replace('massive','')

body = {np.NaN:0,'head':1,'forehead':1,'face':1,'cheeks':1,'facial':1,'jaw':1,'nose':1,'periorbital area':1,
        'abdominal':2,'belly':2,'chest':3,'breast':3,'lungs':3, 'throat':4,'neck':4, 'eye':5,'arm':6,'hand':6,
        'houlder':6,'wrist':6,'leg':7,'hip':7,'feet':7,'calf':7,'ankle':7,'legs':7,'ankles':7,
        'thigh':7,'knee':7,'ear':8, 'ears':8,'urinary tract':9,'testicle':9,'vaginal':9,'vagina':9,'groin':9,'crotch':9,
        'penis':9,'tooth':10,'clavicle':11,'lip':12,'Lips':12,'toenails':13,'rectum':14,'costovertebral angle':19, 
        'mucous membranes':21,'second finger':22,'thumb':22, 'finger':22,'foot':23,'toes':23,'all over':24,'30 %  body':24}

           
def add_median_vital_signal(data_cop):
    lista_parameter=['RR','HR','BP_1','BP_2','T','SPO2 ']
    for x,y in enumerate(data_cop['class']):
        for nr_class in range(1,6):
            if y==nr_class:
                for index,parameter in enumerate(lista_parameter):
                    if str(data_cop.loc[x,parameter])== 'nan':
                        #print(x,y,nr_class,index,parameter,list_val_med[nr_class-1][index])
                        data_cop.loc[x,parameter]=list_median_vital(data_cop)[nr_class-1][index]
                        
def add_median_vital_signal_global(data_cop):
    for x,y in enumerate(data_cop['class']):
        lista_parameter=['RR','HR','BP_1','BP_2','T','SPO2 ']
        list_val_med = list_median_vital_global(data_cop)
        for index,parameter in enumerate(lista_parameter):
             if str(data_cop.loc[x,parameter])== 'nan':
                #print(x,y,index,parameter)
                data_cop.loc[x,parameter]=list_val_med[0][index]

                        
def tag_place(data_cop,body):

    for x,typ in enumerate(data_cop['place ']):
        is_list =0
        if 'left lower quadrant abdominal' in str(typ):
            is_list =1
            data_cop.loc[x,'place '] = data_cop.loc[x,'place '].replace('left lower quadrant abdominal','17 2')
        if 'left lower-quadrant abdominal' in str(typ):
            is_list =1
            data_cop.loc[x,'place '] = data_cop.loc[x,'place '].replace('left lower-quadrant abdominal','17 2')
        if 'stomach' in str(typ):
            is_list =1
            data_cop.loc[x,'place '] = data_cop.loc[x,'place '].replace('stomach','15 2')
        if 'right lower quadrant abdominal' in str(typ):
            is_list =1
            data_cop.loc[x,'place '] = data_cop.loc[x,'place '].replace('right lower quadrant abdominal','18 2')
        if 'left upper quadrant abdominal' in str(typ):
            is_list =1
            data_cop.loc[x,'place '] = data_cop.loc[x,'place '].replace('left upper quadrant abdominal','20 2')
        if 'mid-abdominal' in str(typ):
            is_list =1
            data_cop.loc[x,'place '] = data_cop.loc[x,'place '].replace('mid-abdominal','16')
        if 'low abdominal'  in str(typ):
            is_list =1
            data_cop.loc[x,'place '] = data_cop.loc[x,'place '].replace('low abdominal','17 18 2')
        if 'lower quadrant abdominal' in str(typ):
            is_list =1
            data_cop.loc[x,'place '] = data_cop.loc[x,'place '].replace('lower quadrant abdominal','17 18 2')
        if 'button' in str(typ):
            is_list =1
            data_cop.loc[x,'place '] = data_cop.loc[x,'place '].replace('button','16 2')
        if 'extremitie' in str(typ):
            is_list =1
            data_cop.loc[x,'place '] = data_cop.loc[x,'place '].replace('extremitie','6 7')
        if is_list == 0:
            replace_all( data_cop,'place ',x, body)

breath_words={np.nan:0, 'wheezing':1,'short of breath':2,'no':3,'respiratory arrest':3,'working hard':4,'hard time breathing':4,
              '3n-labored respirations':5,'spontaneously':6,'barely catch':7,'canőt breathe':7, 
              'respiratory distress':8,'9 after coughing':9,'shortness':9,'9 of breath':9, 'increased work of breathing':10,'increasing':11 }

def tag_breath(data_cop, breath_words):   
    match_words(data_cop,'breath',breath_words)
    match_words(data_cop,'breath',{'earlier':'','sudden onset of':'','moderate':'', 'increbodasing':'','after coughing':'','severe':''})

add_inf_words={np.NaN:0,'car accident':1,'vehicle collision':1,'physical activity':2,'accident in water':3,'pain medication not working':4,
           'acetaminophen but it didnőt help':4,'inhaler doesnőt seem to be helping':4,
            'medications not work':4,'changed medications':5,'i feel okay when i lie perfectly still':6,
            'morbidly obese':7,'driving in a car for 12 hours':8,
            'has been sitting in a car for the last 2 days':8,'did not go to dialysis':9,'colitis is acting up':10,'sibling positive strep culture':11,
            'suspicion positive streptococci':11,'large hematoma around the wound':12,
            'ekg - anterior lateral ischemic changes':13,'passing clots the size of oranges':14,'irregular hr':15,
            'obvious chronic obstructive pulmonary disease':16,'requiring insulin occlusion':17}

def tag_additional_inf(data_cop, add_inf_words):    
    add_remove_words(data_cop,'medical history','additional information',['did not go to dialysis'])
    match_words(data_cop,'additional information', add_inf_words)
def clean_place(data_cop):
    stop_words_place = ['left','right','low','lower','bilateral','second','side','of','his','shoulder','fore','xxx',
                       'anterior','s','area','fore','around','the','and','first','belly','sur1']
    pat = '|'.join(r"\b{}\b".format(x) for x in stop_words_place)
    data_cop['place '] = data_cop['place '].str.replace(pat, '')
    data_cop['place '] = data_cop['place '].str.replace('s','').str.replace('fore','').str.replace('thigh','7').str.replace(',','')

    data_cop['place '] = data_cop['place '].str.split()
sym_words={'cough':1,'hoarse':2,'rapid heart rate':3,'funny pressure feeling':4, 'discomfort':4,'sudden headache':5,
           'pressure':4,'sudden onset of a severe headache':5,'foley catheter came out':6,'replacement of her peg tube':7,
           'nausea':8,'fluid buildup':9, 'maroon-colored stool':10,'dark stools':10,'defibrillator gave me a shock':11,'anaphylactic shock':12,
           'strong allergic reaction':12,'tachycardic':13,'sudden onset of palpitations':14,'i couldnőt see about 5 minutes':15,
           'loss of vision':15,'alcohol delirium':16,'feeding tube fell out':17,'sexually assaulted':18,'tingling':19,
           'losing weight':20,'suspected stroke':21,'sudden slurred speech':21,'slurred speech':21,'circulation decreased':22,
           'vaginal bleeding':23,'spotting':23,'sleeping more':24,'hurts to pee':25,'urinate':25,'drooling':26,
           'difficulty swallowing':27,'barely swallow':27,'cold':28,'chills':28,'migraine':29,'food poisoning':30,'cramping':31,
           'crampy':32,'voice is different':33,'dizziness':34,'pneumonia':35,'infection':36,"ran out of adhd medication":37,
           'ran out of my blood 4 medicine':37,'ran out of blood 4 medication':37,'certificate':39,
           'pill for pain':37, 'prescription for pain medication':37,'rabies':50, 'physical exam':39,
           'need tetanus shot':49,'need tetanus booster':49,'need a tetanus shot':49,'runny nose':38,'needs a work note':39,'control visit':39,'pill prevent pregnancy':51,'need  tetanus booster':49,
           'needs to go into detox again':39,'sawdust':40,'bug inside':40,'thing stuck':40,'something':40,'poison ivy':41,'sudden outbursts':42,
           'change in mental status':43,'sneez':44,'hypoglycemia,':45,'good circulation':46,'big':47,'sensation':48
          }

#data.loc[142,'place ']='lungs'
#data.loc[198,'place ']='foot'
#data.loc[171,'type of injury']='maybe bitten'
def tag_main_symptoms(data_cop,sym_words):

    #data_cop.loc[123,'main symptoms'] = np.NaN
    match_words(data_cop,'main symptoms',sym_words)
    word_remove_list={'fever':'','ing up thick green phlegm  xxx':'','ing':'','on one side of my':'','decreased':'','when':''}
    data_cop.loc[113,'main symptoms']=str(data_cop.loc[113,'main symptoms']).replace('fever','')


    match_words(data_cop,'main symptoms',word_remove_list)
    add_remove_words(data_cop,'main symptoms','skin', ['red'])
    data_cop['main symptoms'] = data_cop['main symptoms'].str.replace('pani','pain')
    data_cop['main symptoms'] = data_cop['main symptoms'].str.replace('.','').str.replace(',','')
aw_words = {np.NaN:0,'unresponsive':1,'unrestrained':1,'somnolent':2,'lethargic':2,'listless':2,'weak':13,'weaknes':13,' tiring out':13,
            'not acting right':3,'flaccid':3,'no response to pain':1,
            'gcs 3':1,'unequal pupils':4,'moaning':5,'does not respond to voice':6,'unable to answer':6,'30 %  body surface area':7, 'alert':8,'14 gcs':8,
            'oriented':9,'pass out':17,'moving all extremities':10,'sititng':10,'sitting,':10,'conscious':10,'anxious':11,'sudden outbursts':11,
            'un10 for a couple of minutes':17,'unsteady gait':10,'powerless':13, 'before i 17':13,
            'unable to remember the events':12,'1 min loss of consciousness':17,'1 min loss of 10ness':17,'crying':5,
            'responding appropriately to questions':9,'able to answer your questions':9,'asking':16,
            'responds to verbal stimuli':19,'equal hand grasps':19,'confused':14,'awake':16,'currently aware':16,
            'dis9 to time and place':12,'quiet':15,'cooperative':16,'pleasant':15,'active':16,'looks well':16,
            'running':16,'looking around':10,'remembers the fall':20,'acting appropriately':16,
            'answers your questions appropriately':20, '9 to person and place and time':20,'exhausted':21,
            'sensation decreased':21,'feeling run down':21,'dizzy':22,'despairing':23}

def tag_awareness(data_cop,aw_words):

    match_words(data_cop,'awareness',aw_words)
    data_cop['awareness'] = data_cop['awareness'].str.replace('sudden','').str.replace('13nes','13').str.replace('to painful stimuli','').str.replace('questions','')
    data_cop['awareness'] = data_cop['awareness'].str.replace(',','')
types_words ={np.nan:0,'hit':1,'punched':1,'fell':2,'fall':2,'slipped':2,'lacerations':3,'laceration':3,
    'gunshot':4,'shot':4,'scratched':5,'stab wound':6,'pierce':6, 'stepped on a rusty nail':6,
    'stabbed':6,'cut off':7,'open area':8,'open fracture':9,'burn':10,'broken':20,'abrasion':11,'ragged':11,'injured':11,
    'cut':12,'twisted':13,'maybe bitten':14,'refusing to move':15,'stiff':15,"can't move":15,'flaccid':15,'movement':16,'not 20':17,
    'no obvious deformity':17,'shortened':18,'obvious deformity':18,'obviously deformed':18,'obvious deformity':18,'dislocation':18,'everted':19}


#data.loc[29,'awareness']='flaccid'
#data.loc[29,'type of injury']=np.NaN
#remove_words={'superficial':'','single':'','self-inflicted':'','multiple':''}
#add_remove_words('type of injury','place ',['right hip','right leg','face','left arm'])
def tag_typs_injury(data_cop,types_words):
    for index, typ in enumerate(data_cop['type of injury']): 
        replace_all(data_cop,'type of injury',index,types_words) 
    match_words(data_cop,'type of injury',{'superficial':'','single':'','self-inflicted':'','multiple':'','tree':'',
                                           'externally rotated':'','on the ice':'','11s':'11'})
#tag_typs_injury()
    data_cop['type of injury'] = data_cop['type of injury'].str.replace(',','')
def tag_fever(data_cop):
    data_cop['fever'] = data_cop['fever'].map({np.nan:0,'yes':1,'101.66':1,'103':1,'no':0})

deh_words ={np.NaN:0,'vomiting':1,'vomitig':1,'diarrhea':2,'no pee':3,'nauseous':4}
def tag_dehydration(data_cop,deh_words):
    match_words(data_cop,'dehydration', deh_words)
def tag_appetite(data_cop):
    appetite_words ={np.NaN:0,'no':1,'yes':2}
    match_words(data_cop,'appetite', appetite_words)
def tag_defibrillated(data_cop):
    data_cop['defibrillated ']=data_cop['defibrillated '].replace('no',0).replace(np.nan,0)
def add_val_awareness_4_5(data_cop):
    for x,y in enumerate(data_cop['class']):
        if y==5 and str(data_cop.loc[x,'awareness'])=='nan':
            data_cop.loc[x,'awareness']='alert'
        if y==4 and str(data_cop.loc[x,'awareness'])=='nan':
            data_cop.loc[x,'awareness']='alert'
#add_val_awareness_4_5()
def add_no_if_nan(data_cop):
    name_no = ['CPR','pregnat','defibrillated ','suicide attempt','fever','bleeding','alcohol consumption']
    name_lack_inf = ['medical history','main symptoms','breath','appetite']
    for name in name_no:
        data_cop[name].replace(np.nan, 'no',inplace=True)
    for x in name_lack_inf:
        data_cop[x].replace(np.nan, 0 ,inplace=True)

def tag_pregnat(data_cop):
    data_cop['pregnat'] = data_cop['pregnat'].replace('no',0,).replace('suspicion',2).replace('yes',1)
def drop_cm( data_cop,index,tex,val_int):
    data_cop.loc[index,'type of injury'] = data_cop.loc[index,'type of injury'].replace(tex,'')
    data_cop.loc[index,'type of injury'] = data_cop.loc[index,'type of injury'].replace(val_int,'')

def add_col_typ_size(data_cop):
    lista = []

    for x,y in enumerate(data_cop['type of injury']):
        zero = 1
        if '2-' in str(y):
            drop_cm(data_cop,x,'centimeter','2-')
            lista.append(2)
            zero =0
        if 'cm' in str(y):
            drop_cm(data_cop,x,'cm','2')
            lista.append(2)
            zero =0
        if '3-' in str(y):
            drop_cm(data_cop,x,'centimeter','3-')
            lista.append(3)
            zero =0
        if '6-' in str(y):
            drop_cm(data_cop,x,'centimeter','6-')
            lista.append(6)
            zero =0
        if '4-' in str(y):
            drop_cm(data_cop,x,'centimeter','4-')
            lista.append(4)
            zero =0
        if 'large' in str(y):
            drop_cm(data_cop,x,'large','large')
            lista.append(4)
            zero =0
        if zero==1:
            lista.append(0)
    data_cop.insert(13, "Typ_size", lista, True) 
def drop_fall( data_cop,index,tex):
    data_cop.loc[index,'type of injury'] = data_cop.loc[index,'type of injury'].replace(tex,'')   
def add_col_typ_fall_height(data_cop):
    lista = []
    for x,y in enumerate(data_cop['type of injury']):
        zero = 1
        if '8 feet' in str(y):
            drop_fall(data_cop,x,'8 feet')
            lista.append(8)
            zero =0
        if '5 feet' in str(y):
            drop_fall(data_cop,x,'5 feet')
            lista.append(5)
            zero =0
        if '4 feet' in str(y):
            drop_fall(data_cop,x,'4 feet')
            lista.append(4)
            zero =0
        if '5 m' in str(y):
            drop_fall(data_cop,x,'5 m')
            lista.append(16)
            zero =0
        if zero==1:
            lista.append(0)
    data_cop.insert(14, "Typ_fall_height", lista, True) 
def separate_col_BP(data_cop):  
    lista_BP_1=[]
    lista_BP_2=[]
    for x,y in enumerate(data_cop['BP']):
        if str(y)=='nan':
            lista_BP_1.append(np.nan)
        else:
            lista_BP_1.append(str(y).split('/')[0])
        if str(y)=='nan':
            lista_BP_2.append(np.nan)
        else:
            lista_BP_2.append(str(y).split('/')[1])
    return   lista_BP_1,  lista_BP_2

def drop_add_BP(data_cop,BP_1,BP_2):  
    data_cop.drop('BP',axis='columns', inplace=True)
    data_cop.insert(6, "BP_1", BP_1, True)
    #print(BP_1,BP_2)
    data_cop.insert(7, "BP_2", BP_2, True) 
#separate_col_BP(data)
def clean_col_stop_words(data_cop):
    name_col = ['skin','medical history','main symptoms', 'type of injury', 'additional information','awareness','breath']

    for x in name_col:
        data_cop[x] = data_cop[x].str.lower().str.strip()

        stop_words = ["and","the", "is","his", "from","to","are","my","a",
                      "was",'xxx',"did","for","of","me","on","into","i","nan",
                     "been","has","be","in","does","your","xxxx",'-','drop_me','ice','cat','by','tree',
                     'touch',' oozing', 'around', 'site','sandpaper','very',',']

        pat = '|'.join(r"\b{}\b".format(x) for x in stop_words)
        data_cop[x] = data_cop[x].str.replace(pat, '')
        data_cop[x] = data_cop[x].str.split()

    data_cop['dehydration'].str.lower().str.strip()
    data_cop['dehydration'] = data_cop['dehydration'].str.split(' xxx ')
#clean_col_stop_words()
def uniform_value_pregnat(data_cop): 
    for x, y in enumerate(data_cop['pregnat'].str.contains('was',na=False, regex=True)):
        if y:
            was_pregnat = data_cop['pregnat'].loc[x].replace('was ','-')
            data_cop['pregnat'].loc[x]= was_pregnat
            # print(data['pregnat'].loc[x])
            if 'w' in data_cop['pregnat'].loc[x]:
                value_pregnat = float(data_cop['pregnat'].loc[x].replace('w',''))
                value_pregnat = value_pregnat*7
                data_cop['pregnat'].loc[x] = value_pregnat
            elif 'm' in data_cop['pregnat'].loc[x]:
                value_pregnat = float(data_cop['pregnat'].loc[x].replace('m',''))
                value_pregnat = value_pregnat*30
                data_cop['pregnat'].loc[x] = value_pregnat
            elif 'd' in data_cop['pregnat'].loc[x]:
                value_pregnat = float(data_cop['pregnat'].loc[x].replace('d',''))
                data_cop['pregnat'].loc[x] = value_pregnat

    for x, y in enumerate(data_cop['pregnat'].str.contains('w',na=False, regex=True)):
        if y:
            value_pregnat = float(data_cop['pregnat'].loc[x].replace('w',''))
            value_pregnat = value_pregnat*7
            data_cop['pregnat'].loc[x] = value_pregnat
    for x, y in enumerate(data_cop['pregnat'].str.contains('m',na=False, regex=True)):
        if y:
            value_pregnat = float(data_cop['pregnat'].loc[x].replace('m',''))
            value_pregnat = value_pregnat*30
            data_cop['pregnat'].loc[x] = value_pregnat
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def uniform_value_old(data_cop):   

    data_cop['old'] = data_cop['old'].str.replace('>1m and <=12 m','150 d')

    for x,y in enumerate(data_cop['old']):
        if 'nan' != str(data_cop['old'].loc[x]) and is_number(y):
                #print(x,y)
                val = float(y)
                val= val*365
                data_cop['old'].loc[x] = val
                #print( data['old'].loc[x])
        else:
                if '>'in str(data_cop['old'].loc[x]).strip():
                    data_cop['old'].loc[x] = data_cop['old'].loc[x].replace('>18','7300 d')
                if '<'in str(data_cop['old'].loc[x]):
                    data_cop['old'].loc[x] = data_cop['old'].loc[x].replace('<18','3285 d')
                if 'd'in str(data_cop['old'].loc[x]):
                    data_cop['old'].loc[x] = float(data_cop['old'].loc[x].replace('d',''))
                if 'w'in str(data_cop['old'].loc[x]):
                    value_old = float(data_cop['old'].loc[x].replace('w',''))
                    value_old = value_old*7
                    data_cop['old'].loc[x] = value_old
                if 'm'in str(data_cop['old'].loc[x]):
                    value_old = float(data_cop['old'].loc[x].replace('m',''))
                    value_old = value_old*30
                    data_cop['old'].loc[x] = value_old
    old_int = float(data_cop['old'].mean())
    data_cop['old'] = data_cop['old'].fillna(old_int)
#uniform_value_old():

def vital_signal(data_cop,x,b,C):
    y = data_cop.loc[x,'old']
    if y<=180 and b :
        for c,d in C.items():
            if c is'RR' and d is True:
                data_cop.loc[x,'RR'] = 45
            if c=='SPO2' and d:
                data_cop.loc[x,'SPO2 ']=98
            if c=='BP' and d:
                data_cop.loc[x,'BP']='80/55'
            if c=='HR' and d:
                data_cop.loc[x,'HR']=140
            if c=='T' and d:
                data_cop.loc[x,'T']=98.6
    if y>180 and y<=360 and b:
        for c,d in C.items():
            if c=='RR' and d:
                data_cop.loc[x,'RR'] = 25
            if c=='SPO2' and d:
                data_cop.loc[x,'SPO2 ']=99
            if c=='BP' and d:
                data_cop.loc[x,'BP']='92/60'
            if c=='HR'and d:
                data_cop.loc[x,'HR']=80
            if c=='T' and d:
                data_cop.loc[x,'T']=98.6
    if y>360 and y<=1800 and b:
        for c,d in C.items():
            if c=='RR' and d:
                data_cop.loc[x,'RR'] = 22
            if c=='SPO2' and d:
                data_cop.loc[x,'SPO2 ']=100
            if c=='BP' and d:
                data_cop.loc[x,'BP']='103/70'
            if c=='HR' and d: 
                data_cop.loc[x,'HR']=65
            if c=='T' and d:
                data_cop.loc[x,'T']=98.6
    if y>1800 and y<=3960 and b:
        for c,d in C.items():
            if c=='RR' and d:
                data_cop.loc[x,'RR'] = 13
            if c=='SPO2' and d:
                data_cop.loc[x,'SPO2 ']=98
            if c=='BP' and d:
                data_cop.loc[x,'BP']='91/60'
            if c=='HR' and d: 
                data_cop.loc[x,'HR']=110
            if c=='T' and d:
                data_cop.loc[x,'T']=97.9
    if y>3960 and y<=6840 and b:
        for c,d in C.items():
            if c=='RR' and d:
                data_cop.loc[x,'RR'] = 16
            if c=='SPO2' and d:
                data_cop.loc[x,'SPO2 ']=99
            if c=='BP' and d:
                data_cop.loc[x,'BP']='97/71'
            if c=='HR' and d: 
                data_cop.loc[x,'HR']=66
            if c=='T' and d:
                data_cop.loc[x,'T']=99.1
    if y>6840 and b:
        for c,d in C.items():
            if c=='RR' and d:
                data_cop.loc[x,'RR'] = 16
            if c=='SPO2' and d:
                data_cop.loc[x,'SPO2 ']=98
            if c=='BP' and d:
                data_cop.loc[x,'BP']='120/70'
            if c=='HR' and d : 
                data_cop.loc[x,'HR']=70
            if c=='T' and d:
                data_cop.loc[x,'T']=98.2
def add_value_vital_signal_and_drop_col(data_cop):

    data_cop['vital signal']=data_cop['vital signal'].map({'normal limits':1,'stable':1,'normal limits.':1})
    is_stable = data_cop['vital signal'] == 1

    for x,b in enumerate(is_stable):
        c={'RR':True,'SPO2':True,'BP':True,'HR':True,'T':True}
        vital_signal(data_cop,x,b,c)   
    data_cop.drop('vital signal', 1, inplace = True)
    for a,b in  enumerate(data_cop['RR']):
        if data_cop.loc[a,'class']==5 or data_cop.loc[a,'class']==4 or data_cop.loc[a,'class']==3:
            if str(b) =='nan' and str(data_cop.loc[a,'RR'])=='nan' and str(data_cop.loc[a,'HR'])=='nan' and str(data_cop.loc[a,'BP'])=='nan':
                c={'RR':True,'SPO2':True,'BP':True,'HR':True,'T':True}
                b=1
                vital_signal(data_cop,a,b,c)
            if str(b) =='nan':
                b=1
                c={'RR':True, 'SPO2':False, 'BP':False,'HR':False,'T':False}
                vital_signal(data_cop,a,b,c)
            if str(data_cop.loc[a,'HR'])=='nan':
                b=1
                c={'RR':False,'SPO2':False,'BP':False,'HR':True,'T':False}
                vital_signal(data_cop,a,b,c)
            if str(data_cop.loc[a,'SPO2 '])=='nan':
                b=1
                c={'RR':False, 'SPO2':True, 'BP':False,'HR':False,'T':False}
                vital_signal(data_cop,a,b,c)
            if str(data_cop.loc[a,'BP']):
                b=1
                c={'RR':False,'SPO2':False,'BP':True,'HR':False,'T':False}
                vital_signal(data_cop,a,b,c)
            if str(data_cop.loc[a,'T']):
                b=1
                c={'RR':False,'SPO2':False,'BP':False,'HR':False,'T':True}
                vital_signal(data_cop,a,b,c)
    #print(data_cop['BP'])

#add_value_vital_signal_and_drop_col()  
# print(data_cop['RR'].notnull().sum(),data_cop['BP'].notnull().sum(), data_cop['HR'].notnull().sum(), data_cop['SPO2 '].notnull().sum())
def tag_intubation_EMS_bleeding_alcohol_con_CPR_sex(data_cop):   
    name = ['intubation','EMS','bleeding','alcohol consumption','CPR','sex']
    for x in name:
        data_cop[x] = data_cop[x].map({'no':0,'on':0,'yes':1,'Yes':1,'male':0,'female':1})

    # data['sex'] = data['sex'].replace(to_replace = np.nan, value = 2)  
    pd.to_numeric(data_cop[x])
    data_cop['pain'] = data_cop['pain'].replace(to_replace = np.nan, value = 0) 
    data_cop['when'] = data_cop['when'].replace(to_replace = np.nan, value = 0) 

    data_cop['sex'] = data_cop['sex']. fillna (method = 'ffill')
def change_val_SPO2(data_cop):
    for x,y in enumerate(data_cop['SPO2 ']):
        if y ==" < 90":
            data_cop.loc[x,'SPO2 '] = 89
    data_cop['SPO2 '] = pd.to_numeric(data_cop['SPO2 '])
def uniform_value_when(data_cop):  
    for x,y in enumerate(data_cop['when']):
        if str(y).strip() == "few hours ago":
            data_cop.loc[x,'when']=5*60
        if str(y).strip() == "couple of days ago":
            data_cop.loc[x,'when']=3*24*60
        if str(y).strip() == "early this morning":
            data_cop.loc[x,'when']=4*60
        if str(y).strip() == "few days":
            data_cop.loc[x,'when']=5*24*60
        if str(y).strip() == "this morning":
            data_cop.loc[x,'when']=6*60
        if str(y).strip() == "last night":
            data_cop.loc[x,'when']=12*60
        if str(y).strip() == "today":
            data_cop.loc[x,'when']=3*60
        if str(y).strip() == "1 months":
            data_cop.loc[x,'when']=30*24*60
        if str(y).strip() == "one month":
            data_cop.loc[x,'when']=30*24*60
        if ' h' in str(data_cop.loc[x,'when']):
            value_pregnat = float(data_cop.loc[x,'when'].replace('h',''))
            data_cop['when'].loc[x] = value_pregnat *60
        if 'min' in str(data_cop.loc[x,'when']).strip():
            data_cop['when'].loc[x] = float(data_cop.loc[x,'when'].replace('min',''))
        if 'days' in str(data_cop.loc[x,'when']).strip():
            value_pregnat = float(data_cop.loc[x,'when'].replace('days',''))
            data_cop['when'].loc[x] = value_pregnat *24*60
    data_cop['when'] = pd.to_numeric(data_cop['when'])


# In[4]:


dic_col={'main symptoms':sym_words,'breath':breath_words, 'type of injury':types_words,'medical history':med_his_words,'dehydration':deh_words, 'awareness':aw_words,
         'place ':body,'additional information':add_inf_words,'skin':skin_word}

#utworzenie kolumn binarnych
def add_name(name, long):
    lista_nicosci=[]
    for x in range(0,long+1):
         lista_nicosci.append(str(name)+'_{}'.format(x))
    return lista_nicosci
#add_name('name', 20)

def matrix_as_col(data,dic_long,col_name):
    new_col=[]
    col_name_list =[]
    for index, val in enumerate(data[col_name]):
        vector =np.zeros(dic_long+1).tolist()
        b=0
        for x in range(dic_long+1):
            #print(dic_long)
            #print(col_name_list)
            for a in val:
                #print(val,x)
                if int(a)== int(x):
                    vector[x]=1
                    b=b+1
                    if len(val)== b:
                        new_col.append(vector)
                        #print(col_name,x)
                        #print(vector)
    #data[col_name]=new_col   
    #all_val.append(new_col)
    col_name_list = add_name(col_name, dic_long)
    df=pd.DataFrame(new_col, columns=col_name_list)
    return df

def dataFrame_matrix(clean_data_2):
    df=pd.DataFrame()
    for kay,val in dic_col.items():
        matrix_data =matrix_as_col(clean_data_2,max(val.values()),kay)
        df = pd.concat([df,matrix_data], axis=1)
    return df

def dropna_columns(name_col_dropna,clean_data_2):
    df2 = clean_data_2.drop(name_col_dropna, axis = 1) 
    return df2
def get_name(dic_col):
    name_col_dropna=[]
    for kay,val in dic_col.items():
        name_col_dropna.append(kay)
    return name_col_dropna

def connect_df(df1,df2):
    df_con = pd.concat([df1,df2], axis=1)
    return df_con
    


# In[ ]:





# In[ ]:





# In[ ]:




