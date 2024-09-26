# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 10:01:38 2024

@author: sh032
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
from sklearn.model_selection import train_test_split
import numpy as np




datastore=[]
with open(r"C:\Users\sh032\Sarcasm_Headlines_Dataset.json","r") as file:
    for line in file:
        datastore.append(json.loads(line.strip()))
    
sentences=[]
labels=[]
urls=[]

for item in datastore:
    labels.append(item["is_sarcastic"])
    sentences.append(item["headline"])
    urls.append(item["article_link"])
    


X_train,X_test,y_train,y_test=train_test_split(sentences,labels,test_size=0.2)

vocab_size=1000
padding_type="post"
max_length=100
trunc_type="post"



tokenizer=Tokenizer(num_words=vocab_size, oov_token="<00v>")
tokenizer.fit_on_texts(X_train)
word_index=tokenizer.word_index

sequences_train=tokenizer.texts_to_sequences(X_train)
sequences_test=tokenizer.texts_to_sequences(X_test)



padded_train=pad_sequences(sequences_train,maxlen=max_length,padding=padding_type, truncating="post")
padded_test=pad_sequences(sequences_test,maxlen=max_length,padding=padding_type, truncating=trunc_type)


y_train=np.array(y_train).reshape((len(y_train),1))
y_test=np.array(y_test).reshape((len(y_test),1))


embedding_dim=16

model=keras.Sequential([
    keras.layers.Embedding(vocab_size,embedding_dim,input_length=max_length), #the direction of each word is learnt epoch by epoch
    keras.layers.GlobalAveragePooling1D(), #adding up the vectors
    keras.layers.Dense(24,activation="relu"),
    keras.layers.Dense(1,activation="sigmoid")
    ])

model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])


num_epochs=5
model.fit(padded_train,y_train,epochs=num_epochs,validation_data=(padded_test,y_test),verbose=2)


new_sentence=["granny starting to fear spiders"]
new_sequence=tokenizer.texts_to_sequences(new_sentence)
new_padded=pad_sequences(new_sequence,maxlen=max_length,padding=padding_type, truncating=trunc_type)
prediction=model.predict(new_padded)

print(prediction)