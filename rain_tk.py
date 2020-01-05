
from tkinter import *
import tkinter as tk
from PIL import Image,ImageTk

from pymongo import MongoClient

import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def location():
    dataRead()
    fig=plt.figure(figsize=(13,6))
    ax=fig.add_subplot()
    dfg=df.groupby('SUBDIVISION').mean().sort_values(by='ANNUAL',ascending=False)['ANNUAL']
    dfg.plot(kind='bar',color='r',width=0.5,title='Subdivision wise ANNUAL Rainfall',fontsize=20)
    #print(dfg)
    plt.xticks(rotation = 90,fontsize=10)
    plt.ylabel('Average Annual RainFall(mm)')
    ax.title.set_fontsize(20)
    ax.xaxis.label.set_fontsize(10)
    ax.yaxis.label.set_fontsize(14)
    #print(df.groupby('SUBDIVISION').mean().sort_values(by='ANNUAL',ascending=False)['ANNUAL'][[33,34,35]])
    #print(df.groupby('SUBDIVISION').mean().sort_values(by='ANNUAL',ascending=False)['ANNUAL'])
    #print(str(dfg.max()) + " " + str(dfg.loc[dfg==dfg.max()].index.values))
    #print(str(dfg.min()))    
    #lres=Label(top,font=("Calibi",20,"bold"),fg="white",bg="black",width=60,height=10,text='***Result*** \n\n Max Rainfall: ' + str(dfg.max()) + ' occurred in ' + str(dfg.loc[dfg==dfg.max()].index.values) + "\n" + 'Min Rainfall: ' + str(dfg.min()) + ' occurred in ' + str(dfg.loc[dfg==dfg.min()].index.values) + "\n" + 'Means: ' +str(dfg.mean())).pack()
    
    ltext='***Result*** \n\n Max Rainfall: ' + str(dfg.max()) + ' occurred in ' + str(dfg.loc[dfg==dfg.max()].index.values) + "\n" + 'Min Rainfall: ' + str(dfg.min()) + ' occurred in ' + str(dfg.loc[dfg==dfg.min()].index.values) + "\n" + 'Means: ' +str(dfg.mean())
    labeltext.set(ltext)
    plt.show()

def yearly():
    dataRead()
    fig = plt.figure(figsize=(13,6))
    ax = fig.add_subplot()
    dfg=df.groupby('YEAR').sum()['ANNUAL']
    #print(dfg)
    dfg.plot(kind='line',title='Overall Rainfall in Each Year',fontsize=20)
    
    
    plt.ylabel('Overall Rainfall (mm)')
    ax.title.set_fontsize(20)
    ax.xaxis.label.set_fontsize(15)
    ax.yaxis.label.set_fontsize(15)
    #print(dfg.sort_values(ascending=False))
    #lres=Label(top,font=("Calibi",20,"bold"),fg="white",bg="black",width=45,height=10,text='***Result*** \n\n Max Rainfall: ' + str(dfg.max()) + ' occurred in ' + str(dfg.loc[dfg==dfg.max()].index.values) + "\n" + 'Min Rainfall: ' + str(dfg.min()) + ' occurred in ' + str(dfg.loc[dfg==dfg.min()].index.values) + "\n" + 'Means: ' +str(dfg.mean())).pack()
    
    ltext='***Result*** \n\n Max Rainfall: ' + str(dfg.max()) + ' occurred in ' + str(dfg.loc[dfg==dfg.max()].index.values) + "\n" + 'Min Rainfall: ' + str(dfg.min()) + ' occurred in ' + str(dfg.loc[dfg==dfg.min()].index.values) + "\n" + 'Means: ' +str(dfg.mean())
    labeltext.set(ltext)
   
    
    #print('Max Rainfall: ' + str(dfg.max()) + ' occurred in ' + str(dfg.loc[dfg==dfg.max()].index.values))
    #print('Min Rainfall: ' + str(dfg.min()) + ' occurred in ' + str(dfg.loc[dfg==dfg.min()].index.values))
    #print('Means: ' +str(dfg.mean()))
    plt.show()

def monthly():
    dataRead()
    months=df.columns[2:14]
    #print(months)
    
    fig=plt.figure(figsize=(13,6))
    ax=fig.add_subplot()
    xlb=df['SUBDIVISION'].unique()
    xlb.sort()
    #print(xlb.sort())
    dfg=df.groupby('SUBDIVISION').mean()[months]
    #print(dfg)
    #dfg.plot(kind='bar',width=1.0,title='Overall Rainfall in each Month of the Subdivision',ax=ax,fontsize=20)
    dfg.plot(kind='bar',width=1.0,title='Overall Rainfall in each Month of the subdivision',ax=ax,fontsize=20)
    
    #plt.xticks(np.linspace(0,35,36,endpoint=True),xlb)
    plt.xticks(rotation=90,fontsize=10)
    plt.ylabel('Rainfall (mm)')
    #plt.legend(loc='upper right', fontsize = 'xx-large')
    
    ax.title.set_fontsize(20)
    ax.xaxis.label.set_fontsize(10)
    ax.yaxis.label.set_fontsize(10)
    

    dfg=dfg.mean()[months]
    ltext='***Result*** \n\n Max Rainfall: ' + str(dfg.max()) + ' occurred in ' + str(dfg.loc[dfg==dfg.max()].index.values) + "\n" + 'Min Rainfall: ' + str(dfg.min()) + ' occurred in ' + str(dfg.loc[dfg==dfg.min()].index.values) + "\n" + 'Means: ' +str(dfg.mean())
    labeltext.set(ltext)
    
    #lres=Label(top,font=("Calibi",20,"bold"),fg="white",bg="black",width=45,height=10,text=labeltext).pack()
    #print(dfg)
    #print('Max Rainfall: ' + str(dfg.max()) + ' occurred in ' + str(dfg.loc[dfg==dfg.max()].index.values))
    #print('Min Rainfall: ' + str(dfg.min()) + ' occurred in ' + str(dfg.loc[dfg==dfg.min()].index.values))
    #print('Means: ' +str(dfg.mean()))
    plt.show()
    

def month_yearly():
    dataRead()
    months = df.columns[2:14]
    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(111)
    #xlb=df['YEAR'].unique()
    #xlb.sort()
    dfg=df.groupby('YEAR').mean()[months]
    dfg.plot.line(title='Overall rainfall in each month of year',ax=ax,fontsize=20)
    #plt.xticks(np.linspace(0,35,36,endpoint=True),xlb)
    plt.xticks(rotation=90,fontsize=10)
    plt.ylabel('Rainfall (mm)')
    plt.legend(loc='upper right',fontsize = 10)
    ax.title.set_fontsize(30)
    ax.xaxis.label.set_fontsize(20)
    ax.yaxis.label.set_fontsize(20)
    
    dfg=dfg.mean()[months]
    #print(dfg)
    #print('Max Rainfall: ' + str(dfg.max()) + ' occurred in ' + str(dfg.loc[dfg==dfg.max()].index.values) + "\n" + 'Min Rainfall: ' + str(dfg.min()) + ' occurred in ' + str(dfg.loc[dfg==dfg.min()].index.values) + "\n" + 'Means: ' +str(dfg.mean()))
    #lres=Label(top,font=("Calibi",20,"bold"),fg="white",bg="black",width=45,height=10,text='***Result*** \n\n Max Rainfall: ' + str(dfg.max()) + ' occurred in ' + str(dfg.loc[dfg==dfg.max()].index.values) + "\n" + 'Min Rainfall: ' + str(dfg.min()) + ' occurred in ' + str(dfg.loc[dfg==dfg.min()].index.values) + "\n" + 'Means: ' +str(dfg.mean())).pack()
    ltext='***Result*** \n\n Max Rainfall: ' + str(dfg.max()) + ' occurred in ' + str(dfg.loc[dfg==dfg.max()].index.values) + "\n" + 'Min Rainfall: ' + str(dfg.min()) + ' occurred in ' + str(dfg.loc[dfg==dfg.min()].index.values) + "\n" + 'Means: ' +str(dfg.mean())
    labeltext.set(ltext)
    plt.show()
    


def linear_modelfun():
    dataRead()
    #Linear Regression
    dfs=df.replace(np.nan,0)
    months = dfs.columns[2:14]
    df2=dfs[['SUBDIVISION',months[0],months[1],months[2],months[3]]]
    #print(df2)

    df2.columns = np.array(['SUBDIVISION','x1','x2','x3','x4'])
    #print(df2.columns)

    for k in range(1,9):
        df3 = dfs[['SUBDIVISION',months[k],months[k+1],months[k+2],months[k+3]]]
        #print(df3)
        df3.columns = np.array(['SUBDIVISION','x1','x2','x3','x4'])
        df2 = df2.append(df3)
        #print(df2)
    #print(df2)
    df2.index = range(df2.shape[0])
    #print(df2.index)
    #print(df2)
    df2.drop('SUBDIVISION', axis=1,inplace=True)
    #print(df2)
    msk=np.random.rand(len(df2))<0.8
    #print(msk)

    df_train=df2[msk]
    df_test=df2[~msk]

    #print(len(df_train))
    #print(len(df_test))

    df_train.index = range(df_train.shape[0])
    df_test.index = range(df_test.shape[0])
    #print(df_train.index)
    #print(df_test.index)

    #print(df_train)
    #print(df_test)

    #print(df_train.info())
    #df_train.index = range(df_train.shape[0])
    #df_test.index = range(df_test.shape[0])

    reg=linear_model.LinearRegression()
    reg.fit(df_train.drop('x4',axis=1),df_train['x4'])
    predicted_values = reg.predict(df_train.drop('x4',axis=1))
    residuals = predicted_values-df_train['x4'].values
    #print('MAD (Training Data): ' + str(np.mean(np.abs(residuals))))
    df_res = pd.DataFrame(residuals)
    df_res.columns = ['Residuals']

    fig = plt.figure(figsize=(13,6))
    ax = fig.add_subplot(111)
    df_res.plot.line(title='Different b/w Actual and Predicted (Training Data)', color = 'c', ax=ax,fontsize=20)
    ax.xaxis.set_ticklabels([])
    plt.ylabel('Residual')
    ax.title.set_fontsize(20)
    ax.xaxis.label.set_fontsize(15)
    ax.yaxis.label.set_fontsize(15)

    predicted_values = reg.predict(df_test.drop('x4',axis=1))
    residuals = predicted_values-df_test['x4'].values
    #print('MAD (Test Data): ' + str(np.mean(np.abs(residuals))))
    df_res = pd.DataFrame(residuals)
    df_res.columns = ['Residuals']

    fig = plt.figure(figsize=(13,6))
    ax = fig.add_subplot(111)
    df_res.plot.line(title='Different b/w Actual and Predicted (Test Data)', color='m', ax=ax,fontsize=20)
    ax.xaxis.set_ticklabels([])
    plt.ylabel('Residual')
    ax.title.set_fontsize(20)
    ax.xaxis.label.set_fontsize(15)
    ax.yaxis.label.set_fontsize(15)

    ltext='***Result*** \n\n MAD (Training Data): ' + str(np.mean(np.abs(residuals))) + '\nMAD (Test Data): ' + str(np.mean(np.abs(residuals)))
    labeltext.set(ltext)

    plt.show()
    
      
def analysis():
    if(i.get()==1):
        location()
    elif(i.get()==2):
        yearly()
    
    elif(i.get()==3):
        monthly()
    elif(i.get()==4):
        month_yearly()
    elif(i.get()==5):
        linear_modelfun()
        
    
def overall():
    global top
    global lres
    global labeltext
    top=Toplevel()
    top.geometry('1500x1500')
    top.title("Overall Rainfall Analysis")
    image2=Image.open(r"C:\Users\suchi\Pictures\rain1.jpg")
    photo2=ImageTk.PhotoImage(image2)
    top.configure(background="black")
    global i
    i=IntVar()

    #l1=Label(top,image=photo2).pack()

    r1=Radiobutton(top,text="Average Annual Rainfall in Each Subdivision",value=1,variable=i,bg="black",fg="white",font=("Calibri",29,"bold")).pack()
    r2=Radiobutton(top,text="Total Rainfall in Each Year",value=2,variable=i,bg="black",fg="white",font=("Calibri",29,"bold")).pack()
    
    r3=Radiobutton(top,text="Monthly Rainfalls Plot in subdivision",value=3,variable=i,bg="black",fg="white",font=("Calibri",29,"bold")).pack()
    r4=Radiobutton(top,text="Monthly Rainfalls Plot (Yearwise)",value=4,variable=i,bg="black",fg="white",font=("Calibri",29,"bold")).pack()
    r5=Radiobutton(top,text="Linear Model on Data",value=5,variable=i,bg="black",fg="white",font=("Calibri",29,"bold")).pack()
    
    b1=Button(top,text="SHOW",font=("Calibri",25,"bold"),bg="black",fg="red",command=analysis).pack()

    labeltext=StringVar()
    lres=Label(top,font=("Calibi",20,"bold"),fg="white",bg="black",width=60,height=10,textvariable=labeltext).pack()
    top.configure()
    top.mainloop()

def dataRead():
    global df
    df=pd.read_csv(r'C:\Users\suchi\rainfall in india 1901-2015.csv',low_memory=False)
    
def info():
    global top
    top=Toplevel()
    top.geometry('1500x1500')
    top.title("Overall Rainfall Information")
    top.configure(background="black")
    
    df=pd.read_csv(r'C:\Users\suchi\rainfall in india 1901-2015.csv',low_memory=False)
    subdivs=df['SUBDIVISION'].unique()
    no_of_subdivs=subdivs.size
    #print("Total SubDivisions :"+str(no_of_subdivs))
    #print(subdivs)
    #print(df.columns)
    #print(df.count)

    l1=Label(top,text="Total number of records :\n",bg="black",fg="white",font=("calibri",14,"bold")).pack()
    l=Label(top,text=str(df.count),bg="black",fg="white",font=("calibri",12,"bold")).pack()
    l2=Label(top,text="\nColumns :",bg="black",fg="white",font=("calibri",14,"bold")).pack()
    l3=Label(top,text=df.columns,bg="black",fg="white",font=("calibri",12,"bold")).pack()
    l4=Label(top,text="\nTotal SubDivisions :" + str(no_of_subdivs),bg="black",fg="white",font=("calibri",14,"bold")).pack()
    l5=Label(top,text=subdivs,bg="black",fg="white",width=170,font=("calibri",12,"bold")).pack()
    top.mainloop()
    
def main():   
    window=Tk()
    window.title("Rainfall Analysis")
    window.geometry('1500x1500')
    window.configure(background="Black")

    #photo1=PhotoImage(file=r"C:\Users\suchi\Pictures\rain4.gif")

    image=Image.open(r"C:\Users\suchi\Pictures\rain2.gif")
    photo=ImageTk.PhotoImage(image)
    l1=Label(window,text="Rainfall Data Analysis",bg="black",fg="white",font=("Calibri",29,"bold")).pack()
    b1=Button(window,text="Click Me!",fg="red",image=photo,width=400,height=200,command=overall)
    b1.pack()

    image1=Image.open(r"C:\Users\suchi\Pictures\rain3.gif")
    photo1=ImageTk.PhotoImage(image1)
    l2=Label(window,text="\nOverall Rainfall Information",bg="black",fg="white",font=("Calibri",29,"bold")).pack()
    b2=Button(window,image=photo1,width=400,height=200,command=info)
    b2.pack()

    b2=Button(window,text="QUIT",bg="black",fg="red",font=("Calibri",29,"bold"),command=quit).pack()
    window.mainloop()
    
def mongodb_fun():
    global df
    client=MongoClient("localhost:27017")
    print(client)
    db=client.rainfall_DB
    #r=db.rainfall_in_indiareview.find({}).limit(30)
    db=db.rainfall_in_India
    df=pd.DataFrame(list(db.find()))
    #print(df)
    

if __name__=='__main__':
    main()
