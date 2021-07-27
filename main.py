import pickle
import uvicorn
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer

# 2. Class which describes Bank Notes measurements
class user_tweet(BaseModel):
    user_tweet : str

pickle_in = open("model.pkl","rb")
classifier=pickle.load(pickle_in)
tweets = pd.read_csv("tweets.csv")
app = FastAPI()

@app.get('/')
async def index ():
    return {'Welcome to Sentiment Analysis '}

@app.post('/predict')
def  predict(data : user_tweet):
    text=pd.DataFrame()
    user_text=user_tweet;
    text[0]=user_text
    tweets.append(text)
    tv = TfidfVectorizer()
    X = tv.fit_transform(tweets).toarray()
    pred=classifier.predict(X)

    if(pred[len(X)-1]==1):
        res = str(user_text) + " : Positive"
    else:
        res = str(user_text) + ": Negative"

    return {
        'prediction': res
    }


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
