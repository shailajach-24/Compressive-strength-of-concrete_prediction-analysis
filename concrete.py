#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tkinter as tk 
from tkinter import messagebox,simpledialog,filedialog
from tkinter import *
import tkinter
from imutils import paths
from tkinter.filedialog import askopenfilename

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn import metrics
from sklearn.model_selection import train_test_split,KFold,cross_val_score,GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor


import warnings
warnings.filterwarnings('ignore')


# In[2]:


root= tk.Tk() 
root.title("Indian Air Quality Analysis")
root.geometry("1300x1200")


# In[3]:


def upload_data():
    global data
    data= askopenfilename(initialdir = "Dataset")
    #pathlabel.config(text=train_data)
    text.insert(END,"Dataset loaded\n\n")


# In[4]:


def data():
    global data
    text.delete('1.0',END)
    data= pd.read_csv('concrete_data.csv')
    text.insert(END,"Top FIVE rows of the Dataset\n\n")
    text.insert(END,data.head())
    text.insert(END,"column names\n\n")
    text.insert(END,data.columns)
    text.insert(END,"Total no. of rows and coulmns\n\n")
    text.insert(END,data.shape)


# In[5]:


def statistics():
    text.delete('1.0',END)
    text.insert(END,"Top FIVE rows of the Dataset\n\n")
    text.insert(END,data.head())
    stats=data.describe()
    text.insert(END,"\n\nStatistical Measurements for Data\n\n")
    text.insert(END,stats)
    null=data.isnull().sum()
    text.insert(END,null)


# In[6]:


def train_test():
    text.delete('1.0',END)
    global X,y
    global x_train,x_test,y_train,y_test,X_train,X_test
    text.delete('1.0',END)
    X=data.drop(['concrete_compressive_strength'],axis=1)
    y=data['concrete_compressive_strength']
    X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.2,random_state=0)
    from sklearn.preprocessing import MinMaxScaler
    sc_X = MinMaxScaler()
    x_train = sc_X.fit_transform(X_train)
    x_test = sc_X.transform(X_test)
    text.insert(END,"Train and Test model Generated\n\n")
    text.insert(END,"Total Dataset Size : "+str(len(data))+"\n")
    text.insert(END,"Training Size : "+str(len(x_train))+"\n")
    text.insert(END,"Test Size : "+str(len(x_test))+"\n")
    return x_train,x_test,y_train,y_test,X_train,X_test


# In[7]:


def RF():
    global New_data,data_test
    global x_train,x_test,y_train,y_test
    
    clf = RandomForestRegressor(n_estimators=50, max_features='sqrt')
    clf = clf.fit(x_train, y_train)
    
    predictions = clf.predict(x_test)
    df_output = pd.DataFrame()
    df_output['concreate_strength'] = data['concrete_compressive_strength']
    df_output['predited_concreate_strength'] = pd.DataFrame(predictions)
    df_output[['concreate_strength','predited_concreate_strength']].to_csv('concrete_strength@RF.csv',index=False)
    MAE= metrics.mean_absolute_error(y_test,predictions)
    MSE=metrics.mean_squared_error(y_test,predictions)
    RMS= np.sqrt(metrics.mean_squared_error(y_test,predictions))
    r_square = metrics.r2_score(y_test,predictions)
    text.insert(END,"Error Rate evaluation\n\n")
    text.insert(END,"mean absolute error : "+str(MAE)+"\n")
    text.insert(END,"mean squared error: "+str(MSE)+"\n")
    text.insert(END,"root mean squared error : "+str(RMS)+"\n")
    text.insert(END,"R_square: "+str(r_square)+"\n")
    text.insert(END,"Predicted Values on Test Data: "+str(predictions)+"\n")
    text.insert(END,"\n\nCheck the Project Directory for Submission CSV file\n\n")
    text.insert(END,"@@@------------------Thank You--------------------@@@")

# In[8]:


def LR():
    text.delete('1.0',END)
    lr = LinearRegression()
    lr = lr.fit(x_train, y_train)
    
    predictions = lr.predict(x_test)
    df_output = pd.DataFrame()
    df_output['concreate_strength'] = data['concrete_compressive_strength']
    df_output['predited_concreate_strength'] = pd.DataFrame(predictions)
    df_output[['concreate_strength','predited_concreate_strength']].to_csv('concrete_strength@lr.csv',index=False)
    MAE= metrics.mean_absolute_error(y_test,predictions)
    MSE=metrics.mean_squared_error(y_test,predictions)
    RMS= np.sqrt(metrics.mean_squared_error(y_test,predictions))
    r_square = metrics.r2_score(y_test,predictions)
    text.insert(END,"Error Rate evaluation\n\n")
    text.insert(END,"mean absolute error : "+str(MAE)+"\n")
    text.insert(END,"mean squared error: "+str(MSE)+"\n")
    text.insert(END,"root mean squared error : "+str(RMS)+"\n")
    text.insert(END,"R_square: "+str(r_square)+"\n")
    text.insert(END,"Predicted Values on Test Data: "+str(predictions)+"\n")
    text.insert(END,"\n\nCheck the Project Directory for Submission CSV file\n\n")
    text.insert(END,"@@@------------------Thank You--------------------@@@")

# In[9]:


def KNN():
    text.delete('1.0',END)
    knn = KNeighborsRegressor()
    knn = knn.fit(x_train, y_train)
    
    predictions = knn.predict(x_test)
    df_output = pd.DataFrame()
    df_output['concreate_strength'] = data['concrete_compressive_strength']
    df_output['predited_concreate_strength'] = pd.DataFrame(predictions)
    df_output[['concreate_strength','predited_concreate_strength']].to_csv('concrete_strength@knn.csv',index=False)
    MAE= metrics.mean_absolute_error(y_test,predictions)
    MSE=metrics.mean_squared_error(y_test,predictions)
    RMS= np.sqrt(metrics.mean_squared_error(y_test,predictions))
    r_square = metrics.r2_score(y_test,predictions)
    text.insert(END,"Error Rate evaluation\n\n")
    text.insert(END,"mean absolute error : "+str(MAE)+"\n")
    text.insert(END,"mean squared error: "+str(MSE)+"\n")
    text.insert(END,"root mean squared error : "+str(RMS)+"\n")
    text.insert(END,"R_square: "+str(r_square)+"\n")
    text.insert(END,"Predicted Values on Test Data: "+str(predictions)+"\n")
    text.insert(END,"\n\nCheck the Project Directory for Submission CSV file\n\n")
    text.insert(END,"@@@------------------Thank You--------------------@@@")

# In[10]:


def svr():
    text.delete('1.0',END)
    svr = SVR(kernel='rbf')
    svr = svr.fit(x_train, y_train)
    
    predictions = svr.predict(x_test)
    df_output = pd.DataFrame()
    df_output['concreate_strength'] = data['concrete_compressive_strength']
    df_output['predited_concreate_strength'] = pd.DataFrame(predictions)
    df_output[['concreate_strength','predited_concreate_strength']].to_csv('concrete_strength@svr.csv',index=False)
    MAE= metrics.mean_absolute_error(y_test,predictions)
    MSE=metrics.mean_squared_error(y_test,predictions)
    RMS= np.sqrt(metrics.mean_squared_error(y_test,predictions))
    r_square = metrics.r2_score(y_test,predictions)
    text.insert(END,"Error Rate evaluation\n\n")
    text.insert(END,"mean absolute error : "+str(MAE)+"\n")
    text.insert(END,"mean squared error: "+str(MSE)+"\n")
    text.insert(END,"root mean squared error : "+str(RMS)+"\n")
    text.insert(END,"R_square: "+str(r_square)+"\n")
    text.insert(END,"Predicted Values on Test Data: "+str(predictions)+"\n")
    text.insert(END,"\n\nCheck the Project Directory for Submission CSV file\n\n")
    text.insert(END,"@@@------------------Thank You--------------------@@@")

# In[11]:


def GB():
    text.delete('1.0',END)
    gbr = GradientBoostingRegressor(n_estimators = 50, learning_rate = 0.09, max_depth=5)
    gbr= gbr.fit(x_train, y_train)
    
    predictions = gbr.predict(x_test)
    df_output = pd.DataFrame()
    df_output['concreate_strength'] = data['concrete_compressive_strength']
    df_output['predited_concreate_strength'] = pd.DataFrame(predictions)
    df_output[['concreate_strength','predited_concreate_strength']].to_csv('concrete_strength@gbr.csv',index=False)
    MAE= metrics.mean_absolute_error(y_test,predictions)
    MSE=metrics.mean_squared_error(y_test,predictions)
    RMS= np.sqrt(metrics.mean_squared_error(y_test,predictions))
    r_square = metrics.r2_score(y_test,predictions)
    text.insert(END,"Error Rate evaluation\n\n")
    text.insert(END,"mean absolute error : "+str(MAE)+"\n")
    text.insert(END,"mean squared error: "+str(MSE)+"\n")
    text.insert(END,"root mean squared error : "+str(RMS)+"\n")
    text.insert(END,"R_square: "+str(r_square)+"\n")
    text.insert(END,"Predicted Values on Test Data: "+str(predictions)+"\n")
    text.insert(END,"\n\nCheck the Project Directory for Submission CSV file\n\n")
    text.insert(END,"@@@------------------Thank You--------------------@@@")

# In[ ]:


def input_values():
    text.delete('1.0',END)
    global cement #our 2nd input variable
    cement = float(entry1.get())

    global blast_furnace_slag 
    blast_furnace_slag = float(entry2.get())

    global fly_ash
    fly_ash = float(entry3.get())
    
    global water
    water = float(entry4.get())
    
    global superplasticizer
    superplasticizer = float(entry5.get())
    
    global coarse_aggregate
    coarse_aggregate = float(entry6.get())
    
    global fine_aggregate
    fine_aggregate = float(entry7.get())
    
    global age
    age = float(entry8.get())
    
    list=[[cement, blast_furnace_slag, fly_ash, water, superplasticizer,
       coarse_aggregate, fine_aggregate, age]]
    gbr = GradientBoostingRegressor(n_estimators = 50, learning_rate = 0.09, max_depth=5)
    gbr= gbr.fit(x_train, y_train)
    predictions = gbr.predict(list)
    text.insert(END,"New values are predicted from Gradient Boosting Regressor\n\n")
    text.insert(END,"Predicted concrete_compressive_strength for the New inputs\n\n")
    text.insert(END,predictions)

font = ('times', 14, 'bold')
title = Label(root, text='Predicting Compressive strength of concrete ')  
title.config(font=font)           
title.config(height=2, width=120)       
title.place(x=0,y=5)

font1 = ('times',13 ,'bold')
button1 = tk.Button (root, text='Upload Data',width=13,command=upload_data) 
button1.config(font=font1)
button1.place(x=60,y=100)

button2 = tk.Button (root, text='Data',width=13,command=data)
button2.config(font=font1)
button2.place(x=60,y=150)

button3 = tk.Button (root, text='statistics',width=13,command=statistics)  
button3.config(font=font1)
button3.place(x=60,y=200)



button5 = tk.Button (root, text='Train & Test',width=13,command=train_test)
button5.config(font=font1) 
button5.place(x=60,y=250)

title = Label(root, text='Application of ML models')
#title.config(bg='RoyalBlue2', fg='white')  
title.config(font=font1)           
title.config(width=25)       
title.place(x=250,y=70)

button6 = tk.Button (root, text='RFT',width=15,bg='pale green',command=RF)
button6.config(font=font1) 
button6.place(x=300,y=100)

button7 = tk.Button (root, text='LR',width=15,bg='sky blue',command=LR)
button7.config(font=font1) 
button7.place(x=300,y=150)

button8 = tk.Button (root, text='KNN',width=15,bg='orange',command=KNN)
button8.config(font=font1) 
button8.place(x=300,y=200)

button9 = tk.Button (root, text='SVR',width=15,bg='violet',command=svr)
button9.config(font=font1) 
button9.place(x=300,y=250)

button10 = tk.Button (root, text='GBR',width=15,bg='violet',command=GB)
button10.config(font=font1) 
button10.place(x=300,y=300)


title = Label(root, text='Enter Input values for the New Prediction')
title.config(bg='black', fg='white')  
title.config(font=font1)           
title.config(width=40)       
title.place(x=60,y=380)

font3=('times',9,'bold')
title1 = Label(root, text='*You Should enter scaled values between 0 and 1')
 
title1.config(font=font3)           
title1.config(width=40)       
title1.place(x=50,y=415)

def clear1(event):
    entry1.delete(0, tk.END)

font2=('times',10)
entry1 = tk.Entry (root) # create 1st entry box
entry1.config(font=font2)
entry1.place(x=60, y=450,height=30,width=150)
entry1.insert(0,'cement')
entry1.bind("<FocusIn>",clear1)

def clear2(event):
    entry2.delete(0, tk.END)

font2=('times',10)
entry2 = tk.Entry (root) # create 1st entry box
entry2.config(font=font2)
entry2.place(x=150, y=450,height=30,width=150)
entry2.insert(0,'blast_furnace_slag')
entry2.bind("<FocusIn>",clear2)


def clear3(event):
    entry3.delete(0, tk.END)

font2=('times',10)
entry3 = tk.Entry (root) # create 1st entry box
entry3.config(font=font2)
entry3.place(x=300, y=450,height=30,width=150)
entry3.insert(0,'fly_ash')
entry3.bind("<FocusIn>",clear3)

def clear4(event):
    entry4.delete(0, tk.END)

font2=('times',10)
entry4 = tk.Entry (root) # create 1st entry box
entry4.config(font=font2)
entry4.place(x=60, y=500,height=30,width=150)
entry4.insert(0,'water')
entry4.bind("<FocusIn>",clear4)

def clear5(event):
    entry5.delete(0, tk.END)

font2=('times',10)
entry5 = tk.Entry (root) # create 1st entry box
entry5.config(font=font2)
entry5.place(x=150, y=500,height=30,width=150)
entry5.insert(0,'superplasticizer')
entry5.bind("<FocusIn>",clear5)

def clear6(event):
    entry6.delete(0, tk.END)

font2=('times',10)
entry6 = tk.Entry (root) # create 1st entry box
entry6.config(font=font2)
entry6.place(x=300, y=500,height=30,width=150)
entry6.insert(0,'coarse_aggregate',)
entry6.bind("<FocusIn>",clear6)

def clear7(event):
    entry7.delete(0, tk.END)

font2=('times',10)
entry7 = tk.Entry (root) # create 1st entry box
entry7.config(font=font2)
entry7.place(x=60, y=550,height=30,width=150)
entry7.insert(0,'fine_aggregate ')
entry7.bind("<FocusIn>",clear7)

def clear8(event):
    entry8.delete(0, tk.END)

font2=('times',10)
entry8 = tk.Entry (root) # create 1st entry box
entry8.config(font=font2)
entry8.place(x=150, y=550,height=30,width=150)
entry8.insert(0,'age')
entry8.bind("<FocusIn>",clear8)



Prediction = tk.Button (root, text='Prediction',width=15,fg='white',bg='green',command=input_values)
Prediction.config(font=font1) 
Prediction.place(x=180,y=600)



font1 = ('times', 11, 'bold')
text=Text(root,height=32,width=90)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set,xscrollcommand=scroll.set)
text.place(x=550,y=70)
text.config(font=font1)

root.mainloop()


    

