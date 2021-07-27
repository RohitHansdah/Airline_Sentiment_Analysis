import pickle
import uvicorn
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from sklearn.feature_extraction.text import TfidfVectorizer

pickle_in = open("model.pkl","rb")
classifier=pickle.load(pickle_in)
tweets = pd.read_csv("tweets.csv")
app = FastAPI()



@app.get('/')
def  predict():
    text=pd.DataFrame()
    user_text="airline was good";
    text[0]=user_text
    tweets.append(text)
    tv = TfidfVectorizer()
    X = tv.fit_transform(tweets).toarray()
    pred=classifier.predict(X)
    if(pred[len(X)-1]==1):
        return { user_text + " : Positive"}
    else:
        return { user_text + ": Negative"}



if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
