from tkinter import *
import sys, os

import nltk
from nltk.tokenize import RegexpTokenizer, word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import classification_report,confusion_matrix, accuracy_score

df = pd.read_csv('spam.csv' , encoding='ISO-8859-1') #read csv file and import it in dataframe format
    
data = df.to_numpy() #convert the pandas Series to NumPy ndarray representing the values in given Series or Index

X = data[:, 1] #X stores text mails
y = data[:, 0] #y stores label of text mails

print(X.shape, y.shape)

tokenizer = RegexpTokenizer('\w+') #create reference variable for class RegexpTokenizer
nltk.download('stopwords') #stopwords in nltk are the most common words in data
sw = set(stopwords.words('english')) #ignore the stop words in english
ps = PorterStemmer() #used to remove the suffixes from an English word and obtain its stem

def getStem(review):
    review = review.lower() #returns lowercased string from given string 
    tokens = tokenizer.tokenize(review) #splits string into substrings of words
    removed_stopwords = [w for w in tokens if w not in sw] #stopwords are removed 
    stemmed_words = [ps.stem(token) for token in removed_stopwords] #tokens  are reduced into root form
    clean_review = ' '.join(stemmed_words) #join all stemmed words to single string 
    return clean_review

# get a clean document
def getDoc(document):
    d = []
    for doc in document:
        d.append(getStem(doc))
    return d

stemmed_doc = getDoc(X)

#print(stemmed_doc[:10])

cv = CountVectorizer() #Create a Vectorizer Object. It is used to transform a given text into a vector on the basis of the frequency (count) of each word that occurs in the entire text.

# create my vocab 
vc = cv.fit_transform(stemmed_doc) #fit_transform() is used on the training data to scale the training data and learn the scaling parameters of that data. Here, the model built will learn the mean and variance of the features of the training set. These learned parameters are then used to scale our test data.

X = vc.todense() #return dense representation of NDFrame

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42) #Split arrays into random train and test subsets

model = MultinomialNB() #create reference variable for multinomail naive bayes classifier. It is suitable for classification with discrete features (e.g., word counts for text classification)
model.fit(X_train, y_train) # estimate the attributes out of the input data and store the model attributes and finally return the fitted estimator
#print(model.score(X_test, y_test))

print("Evaluating the model on training data set")

pred = model.predict(X_train) #perform a prediction for each test instance using the learned parameters by fit()
print(classification_report(y_train ,pred )) #used to measure the quality of predictions from the classification algorithm 
print('Confusion Matrix: \n',confusion_matrix(y_train,pred))
print()
print('Accuracy: ', accuracy_score(y_train,pred)) #computes subset accuracy: the set of labels predicted for a sample must exactly match the corresponding set of labels in y_true

print("Evaluating the model on test data set")

pred = model.predict(X_test)
print(classification_report(y_test ,pred ))
print('Confusion Matrix: \n', confusion_matrix(y_test,pred))
print()
print('Accuracy: ', accuracy_score(y_test,pred))

class MainWindow():

    def __init__(self, master, title, size):
        self.master = master
        self.title = title
        self.size = size
        self.master.title(self.title)
        self.master.geometry(self.size)
        
        self.T=Text(self.master,height=20,width=40)

        self.T.insert(END,"Enter the mail")
        
        self.l=Label(self.master,text="Spam Mail Detection")
        self.l.config(font=("Sans Serif",16))
        
        self.l.pack(pady=10)

        self.T.pack()

        self.productButton = Button(self.master,
                             text="Check",
                                width=20,
                                command=self.productButtonClicked).place(x=85, y=400)

        self.quitMainWindow = Button(self.master,
                                 text="Exit",
                                 width=20,
                                 command=self.on_cancel).place(x=265, y=400)

    
    def productButtonClicked(self):
        #productWindow = Toplevel()
        obj = ProductMenuWindow(self, "Result", "300x300")
        #productFenster = ProductMenuWindow(productWindow,)

    def on_cancel(self):
        print("Success")
        self.master.destroy()        


class ProductMenuWindow(Toplevel):

    def __init__(self, parent, title, size):
        super().__init__(name='product_main_menu')

        self.parent = parent

        self.title(title)

        self.size = size

        self.geometry(size)

        self.configure(bg="black")

        messages = []
        messages.append(str(parent.T.get(1.0,"end-1c")))
    
        def prepare(messages):
            d = getDoc(messages)
            # dont do fit_transform!! it will create new vocab.
            return cv.transform(d) #standard procedure to scale data while building a machine learning model so that the model is not biased towards a particular feature of the dataset and at the same time prevents the model to learn the features/values/trends of our test data
    
        messages = prepare(messages)
    
        y_pred = model.predict(messages)

        self.T=Text(self,height=5,width=25) #used where a user wants to insert multiline text fields 
        
        self.l=Label(self,text="Result") #used to implement display boxes where one can place text or images

        self.l.config(font=("Sans Serif",16))

        self.l.config(text="Result")
        
        self.l.pack( pady=10) #vertical padding

        self.T.insert(END,y_pred[0]) #insert string at end of widget

        self.T.pack() #packs widgets in rows or columns

        self.gobackButton = Button(self,
                               text="Go back to main window",
                               width=20,
                               command=self.on_cancel).place(x=55, y=175) #create the button

    def on_cancel(self):
        self.parent.T.delete("1.0",END) #delete content from text widget 
        self.parent.T.insert(END,"Enter the mail")
        self.destroy()

if __name__ == "__main__":
    mainWindow = Tk()
    mainFenster = MainWindow(mainWindow, "Internship", "500x500")
    mainWindow.configure(bg="black") #used to access an object's attributes after its initialisation.
    mainWindow.mainloop()