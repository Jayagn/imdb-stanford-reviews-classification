# import required packages


# YOUR IMPLEMENTATION
# Thoroughly comment your code to make it easy to follow

import sys
import os
import json
import io
import re
import numpy as np
from keras_preprocessing.text import tokenizer_from_json
from keras.preprocessing.sequence import pad_sequences
from tensorflow import keras
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.layers import Dense, Embedding, Dropout, Flatten


if __name__ == "__main__": 
#    test = sys.argv[1



	# 1. Load your saved model
    
    
    
    imdb_folder = "aclImdb"   
    def get_reviews(data_folder="/test"):
        reviews = []
        labels = []
        for index,sentiment in enumerate(["/neg/", "/pos/"]):
            path = imdb_folder + data_folder + sentiment
            for filename in sorted(os.listdir(path)):
                with open(path + filename, 'r',encoding='utf8') as f:
                    review = f.read()
                    review = review.lower()
                    review = review.replace("<br />", " ")
                    review = re.sub(r"[^a-z ]", " ", review)
                    review = re.sub(r" +", " ", review)
                    review = review.split(" ")
                    reviews.append(review)
                    
                    label = [0, 0]
                    label[index] = 1
                    labels.append(label)
                    
        return reviews, np.array(labels)
    
    test_reviews, test_labels = get_reviews()
    #print(len(test_reviews))
    #print(test_reviews[0])
    #print(test_labels[0])
    
    tokenizer_obj = Tokenizer()
    tokenizer_obj.fit_on_texts(test_reviews)
    test_sequences = tokenizer_obj.texts_to_sequences(test_reviews)
    max_length_arr = [len(s) for s in (test_sequences)]
    max_length = max(max_length_arr)
    test_padded = pad_sequences(test_sequences, maxlen=2495, padding='post', truncating='post')
     
    
    model = '20812666_NLP_model.model'   
    loaded_model = keras.models.load_model(model)
    print(loaded_model)
    print("Predicting")
    print(loaded_model.predict_classes(test_padded))
    print("Evaluating model")
    loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    score = loaded_model.evaluate(test_padded, test_labels, verbose=0)
    print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
    
    
    	# 2. Load your testing data
    
    	# 3. Run prediction on the test data and print the test accuracy