import numpy as np 
import random 
import json

from preprocesing import rem_stopword,remove_punctions,tokenize,stemming,lemmatizing,bag_of_words

from model import my_model

from tensorflow.keras.optimizers import Adam,SGD

if __name__ == "__main__":
    with open("data.json") as json_data:
        datas = json.load(json_data)

    words = []
    classes =[]
    document = []

    dataset = []



    for data in datas['data']:
        for pattern in data['patterns']:

            pattern_nopunc=remove_punctions(pattern.lower())

            pattern_tokenize=tokenize(pattern_nopunc)

            pattern_nostopword = rem_stopword(pattern_tokenize)

            stem_pattern = lemmatizing(pattern_nostopword)

            #lema_pattern = lemmatizing(pattern_nostopword)
            
            #get all preprocessed words in separate list, needed for making bag of words
            words.extend(stem_pattern)

            #get documents i.e. words along with their respective tag
            document.append((stem_pattern,data['tag']))

            #get all the classes i.e. tag for doing one hot encoding of labels i.e. tag
            if data['tag'] not in classes:
                classes.append(data['tag'])

    #print(document)
    words = sorted(list(set(words)))

    classes = sorted(list(set(classes)))

    #assigning empty list of doing one hot encoding of labels

    output_empty = [0] * len(classes)
    for docs in document:
        
        pat = docs[0]

        bag=bag_of_words(pat,words)

        output_row = list(output_empty)

        output_row[classes.index(docs[1])] = 1

        dataset.append([bag,output_row])


    random.shuffle(dataset)

    dataset = np.array(dataset)

    train_x = np.array(list(dataset[:,0]))

    train_y = np.array(list(dataset[:,1]))
    

    #print (len(document), "documents")
    print (len(classes), "classes", classes)
    print (len(words), "unique stemmed words", words)

    
    model = my_model(len(train_x[0]),len(train_y[0]))

    print (model.summary())

    opt = Adam(learning_rate=0.01)
    model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])

    model.fit(train_x,train_y,epochs=100,verbose=1,batch_size=5)

    # saving the model
    save_dir = "/results/"
    model_name = 'keras_mnist.h5'
    model.save(model_name)
    model_path = save_dir + model_name
    print('Saved trained model at %s ' % model_path)

    # save all of our data structures
    import pickle
    pickle.dump( {'words':words, 'classes':classes, 'train_x':train_x, 'train_y':train_y}, open( "training_data", "wb" ) )
