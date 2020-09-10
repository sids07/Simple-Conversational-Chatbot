import random
import json
from tensorflow import keras

import pickle
import numpy as np
from preprocesing import remove_punctions, tokenize, rem_stopword, stemming, bag_of_words,lemmatizing

if __name__=="__main__":
    data = pickle.load( open( "training_data", "rb" ) )
    words = data['words']
    classes = data['classes']   
    train_x = data['train_x']
    train_y = data['train_y']

    with open("data.json") as json_data:
        datas = json.load(json_data)

    model= keras.models.load_model('keras_mnist.h5')

    model.summary()
    print(train_x.shape)
    bot_name = "Sid"
    print("Let's chat! (type 'quit' to exit)")
    while True:
        # sentence = "do you use credit cards?"
        sentence = input("You: ")
        if sentence == "quit":
            break

        sentence = sentence.lower()
        sentence1 = remove_punctions(sentence)

        sentence2 = tokenize(sentence1)

        sentence3 = rem_stopword(sentence2)

        sentence4 = lemmatizing(sentence3)

        inp = bag_of_words(sentence,words)

        inp= np.array(inp).reshape(-1,45)
    
        
        result = model.predict(inp)

        prediction = np.argmax(result)

        tag = classes[prediction]

        prob = result[0][prediction]

        if prob>0.75:
            for data in datas['data']:
                if tag == data['tag']:
                    print("{} : {}".format(bot_name,random.choice(data['response'])))

        else:
            print("{}: I dont understand..".format(bot_name))    

        