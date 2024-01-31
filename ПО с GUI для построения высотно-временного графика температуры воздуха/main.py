

from PyQt5 import uic
from PyQt5.QtWidgets import QApplication
Form, Window = uic.loadUiType("MT.ui")
app = QApplication([])
window = Window()
form = Form()
form.setupUi(window)
window.show()
#app.exec_()

#%%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
def upload_day():
    day = form.dateEdit.dateTime().toString("yyyyMMdd")
    from_directory = form.lineEdit_3.text()
    with open(rf'{from_directory}0mtp{day}.txt') as old, open(r"C:\Users\kirit\Downloads\archive\new.txt", 'w+') as new:
        lines = old.readlines()
    
        col = lines[25]
        new.writelines(lines[25:])
    
    

#%%
def make_df():
    df = pd.read_csv(r"C:\Users\kirit\Downloads\archive\new.txt",  sep="\t")

    #print(df.iloc[:, 1:])
  
    def del_day(time):
        t = time.split(" ")[1]
        return t[:-3]

    def comma_to_dot(nums):

        return float(nums.replace(",","."))
    df.iloc[:,0] = np.vectorize(del_day)(df.iloc[:,0])
    df.iloc[:, 1:] = np.vectorize(comma_to_dot)(df.iloc[:, 1:])
    df = df.set_index("data time")
    return df
#df.head()
#%%%%%%%%%%%
def make_plot():
    

    upload_day()
    df = make_df()
    ###########


    high = form.lineEdit_for_high.text().replace(" ","").split(",")
    print(high)

    x = form.lineEdit.text()
    y = form.lineEdit_2.text()
    
    plt.figure(figsize=(20,10))
    for i in high:
        plt.plot(df.index, df[i] )
    plt.legend(high)
    plt.show()
    print("good")
    
    print()
    
    
    

#%%

form.make_plot_button.clicked.connect(make_plot)



app.exec_()