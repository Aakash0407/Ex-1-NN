<H3>ENTER YOUR NAME : AAKASH P</H3>
<H3>ENTER YOUR REGISTER NO : 212222110001</H3>
<H3>EX. NO.1</H3>
<H3>DATE : 22/08/2024</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```python
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

df=pd.read_csv('/content/Churn_Modelling.csv')
print(df.head())

X=df.iloc[:,:-1].values
print(X)

y=df.iloc[:,-1].values
print(y)

print(df.isnull().sum())

df.duplicated()

df.describe()

df = df.drop(['Surname', 'Geography','Gender'], axis=1)
df.head()

scaler=MinMaxScaler()
df1=pd.DataFrame(scaler.fit_transform(df))
print(df1)

X_train ,X_test ,y_train,y_test=train_test_split(X,y,test_size=0.2)

print(X_train)
print(len(X_train))

print(X_test)
print(len(X_test))
```

## OUTPUT:
### DATASET:
![1](https://github.com/user-attachments/assets/e0b7f894-6e1c-41bd-a2f7-5500c64cd7c3)

### X VALUES:
![2](https://github.com/user-attachments/assets/edd710ab-8d4a-47af-9f42-88040ca7d06c)

### Y VALUES
![3](https://github.com/user-attachments/assets/de23d29a-f09f-424c-a9e7-67f2df5d5f46)

### NULL VALUES:
![4](https://github.com/user-attachments/assets/58f75e03-8173-4bf7-9a02-dc698ccc1473)

### DUPLICATED VALUES:
![5](https://github.com/user-attachments/assets/e50a9a3d-10fd-42d3-a17a-794b32ad7979)

### DESCRIPTION:
![6](https://github.com/user-attachments/assets/5400cd89-ce2b-4542-827f-e96a574350ff)

### NORMALIZED DATASET:
![8](https://github.com/user-attachments/assets/7701bf7c-edcf-4380-b268-b2568bf903fc)

### TRAINING DATASET:
![9](https://github.com/user-attachments/assets/e896cbdc-829e-456c-82da-406eb8a9a7db)

### TESTING DATASET:
![10](https://github.com/user-attachments/assets/e397ac0d-89de-4e2b-93de-53207c874773)

## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


