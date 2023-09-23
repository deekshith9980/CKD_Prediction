

import pandas as pd #used for data manipulation
import numpy as np #used for numerical analysis
from collections import Counter as c #returns counts of classes
import matplotlib.pyplot as plt #used for data visualization
import seaborn as sns #used for data visualization
import missingno as msno #finding missing values
from sklearn.metrics import accuracy_score, confusion_matrix #model performance
from sklearn.model_selection import train_test_split #Splits data in random train and test array
from sklearn.preprocessing import LabelEncoder #encoding the levels of categorical featues
from sklearn.linear_model import LogisticRegression #Classification ML  Algorithm
import pickle #python object hirearchy is converted into a byte stream


# Loading the dataset
data = pd.read_csv(r"D:/mpcopy/Chronic-Kidney-Disease-Detection-Using-Machine-Learning-main/Datasets/chronickidneydisease.csv")

# returns first 10 rows
data.head(10)
#Drop is used for dropping the column
data.drop(['id'],axis=1,inplace=True)
# <h2>Renaming the columns</h2>
#return all the column names
data.columns
#manually giving the names of the columns
data.columns = ['age','blood_pressure','specific_gravity','albumin','sugar','red_blood_cells','pus_cell',
                'pus_cell_clumps','bacteria','blood glucose random','blood_urea','serum_creatinine','sodium','potassium','hemoglobin','packed_cell_volume','white_blood_cell_count','red_blood_cell_count','hypertension','diabetesmellitus','coronary_artery_disease','appetite','pedal_edema','anemia','class']
data.columns
#info will give the summary of the dataset
data.info()

#find the unique elements of the array
data['class'].unique()


#replace is used of renaming
data['class'] = data['class'].replace("ckd\t","ckd")
data['class'].unique()
#only fetch the object  type columns
catcols = set(data.dtypes[data.dtypes=='O'].index.values)
print(catcols)
for i in catcols:
    print("Columns :",i)
    print(c(data[i])) #using counter for checking the no of classes in the column
    print('*'*120+'\n')


#remove is used for removing the column
catcols.remove('red_blood_cell_count')
catcols.remove('packed_cell_volume')
catcols.remove('white_blood_cell_count')
print(catcols)

# only fetch the float and int type columns
contcols = set(data.dtypes[data.dtypes!='O'].index.values)
print(contcols)
for i in contcols:
    print("Continous Columns:",i)
    print(c(data[i]))#using counter for checking the number of classes in the column
    print('*'*120+'\n')

contcols.remove('specific_gravity')
contcols.remove('albumin')
contcols.remove('sugar')
print(contcols)

#using add we can add columns
contcols.add('red_blood_cell_count')
contcols.add('packed_cell_volume')
contcols.add('white_blood_cell_count')
print(contcols)
# <h1>Adding columns which we found categorical</h1>

catcols.add('specific_gravity')
catcols.add('albumin')
catcols.add('sugar')
print(catcols)
# <h1>Rectifying the categorical column classes</h1>
data['coronary_artery_disease']=data.coronary_artery_disease.replace('\tno','no')
c(data['coronary_artery_disease'])
data['diabetesmellitus']=data.diabetesmellitus.replace(to_replace={'\tno':'no','\tyes':'yes',' yes':'yes'})
c(data['diabetesmellitus'])
# <h1>Null Values</h1>
# it will return if any null vales values present
data.isnull().any()
#returns the count
data.isnull().count()
data.packed_cell_volume = pd.to_numeric(data.packed_cell_volume , errors='coerce')
data.red_blood_cell_count = pd.to_numeric(data.red_blood_cell_count  , errors='coerce')
data.white_blood_cell_count = pd.to_numeric(data.white_blood_cell_count , errors='coerce')
# <h1>Handling Continous/Numerical Columns Missing Values</h1>
data['blood glucose random'].fillna(data['blood glucose random'].mean(),inplace = True)
data['blood_pressure'].fillna(data['blood_pressure'].mean(),inplace = True)
data['blood_urea'].fillna(data['blood_urea'].mean(),inplace = True)
data['hemoglobin'].fillna(data['hemoglobin'].mean(),inplace = True)
data['packed_cell_volume'].fillna(data['packed_cell_volume'].mean(),inplace = True)
data['potassium'].fillna(data['potassium'].mean(),inplace = True)
data['red_blood_cell_count'].fillna(data['red_blood_cell_count'].mean(),inplace = True)
data['serum_creatinine'].fillna(data['serum_creatinine'].mean(),inplace = True)
data['sodium'].fillna(data['sodium'].mean(),inplace = True)
data['white_blood_cell_count'].fillna(data['white_blood_cell_count'].mean(),inplace = True)

data['age'].fillna(data['age'].mode()[0], inplace=True)
data['specific_gravity'].fillna(data['specific_gravity'].mode()[0], inplace=True)
data['albumin'].fillna(data['albumin'].mode()[0], inplace=True)
data['sugar'].fillna(data['sugar'].mode()[0], inplace=True)
data['red_blood_cells'].fillna(data['red_blood_cells'].mode()[0], inplace=True)
data['pus_cell'].fillna(data['pus_cell'].mode()[0], inplace=True)
data['pus_cell_clumps'].fillna(data['pus_cell_clumps'].mode()[0], inplace=True)
data['bacteria'].fillna(data['bacteria'].mode()[0], inplace=True)
data['blood glucose random'].fillna(data['blood glucose random'].mode()[0], inplace=True)
data['hypertension'].fillna(data['hypertension'].mode()[0], inplace=True)
data['diabetesmellitus'].fillna(data['diabetesmellitus'].mode()[0], inplace=True)
data['coronary_artery_disease'].fillna(data['coronary_artery_disease'].mode()[0], inplace=True)
data['appetite'].fillna(data['appetite'].mode()[0], inplace=True)
data['pedal_edema'].fillna(data['pedal_edema'].mode()[0], inplace=True)
data['anemia'].fillna(data['anemia'].mode()[0], inplace=True)
data['class'].fillna(data['class'].mode()[0], inplace=True)
data.isna().sum()
# <h1>Label encoding</h1>
#importing label encoding from sklearn
from sklearn.preprocessing import LabelEncoder

for i in catcols:  #looping through all the categorical columns
    print("LABEL ENCODING OF :",i)
    LEi = LabelEncoder() #creating an object of label encoder
    print(c(data[i]))  #getting the classes values before transformation
    data[i] = LEi.fit_transform(data[i]) #transforming our text classes to numerical values
    print(c(data[i]))  #geting class values after transformation
    print('*'*100)

features_name = ['blood_urea','blood glucose random','coronary_artery_disease','anemia','pus_cell',
    'red_blood_cells','diabetesmellitus','pedal_edema']
x = pd.DataFrame(data, columns = features_name)
y = pd.DataFrame(data, columns = ['class'])
print(x.shape)
print(y.shape)
data.isna().sum()
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=2)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
from sklearn.linear_model import LogisticRegression

lgr = LogisticRegression(solver='lbfgs', max_iter=1000)

lgr.fit(x_train.values,y_train.values.ravel())
y_pred = lgr.predict(x_test)

def lgpredict(p1,p2,p3,p4,p5,p6,p7,p8):
    x_new=np.array([p1,p2,p3,p4,p5,p6,p7,p8])
    x_new=x_new.reshape(1,8)
    ll_predict= lgr.predict(x_new)
    y_pred=ll_predict
    return int(ll_predict)
# y_pred1 =lgr.predict([[90,157,1,0,0,1,1,1]])

#print(llpred)
c(y_pred)
accuracy_score(y_test,y_pred)
# <h1>Confusion matrix of our model</h1>
conf_mat = confusion_matrix(y_test,y_pred)
conf_mat
pickle.dump(lgr, open('CKD.pkl','wb'))
