import os
import tensorflow as tf
from sklearn.metrics import  f1_score
import time
import random
import logging
import preprocess_heart_disease_data


# Turn off tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}


def MyDNN(trainX_fully_preprocessed, trainY_binary, testX_fully_preprocessed, testY_binary, hidden_units = [10,5], dropout=0.5,activation_fn=tf.nn.elu, steps=1000 ):

    Model_01 = tf.contrib.learn.DNNClassifier(hidden_units = hidden_units,
                                              n_classes=2,
                                              dropout=dropout,
                                              feature_columns=tf.contrib.learn.infer_real_valued_columns_from_input(trainX_fully_preprocessed),
                                              activation_fn=activation_fn
                                              ,gradient_clip_norm=0.9
                                              ,optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
                                              )
    #This line makes the tf model compatible with the sklearn way of codding
    Model_01 = tf.contrib.learn.SKCompat(Model_01)
    
    start_train = time.time()
    Model_01.fit(trainX_fully_preprocessed,
                 trainY_binary,
                 batch_size=100,
                 steps=steps
                 )

    end_train = time.time()
    
    train_time = end_train - start_train 
    
    test_predict_01 = Model_01.predict(testX_fully_preprocessed)['classes']#The TensorFlow predict object is a bit different from the sklearn object

    start_test = time.time()
    print(start_test)
    f1 = f1_score(testY_binary, test_predict_01)
    end_test = time.time()
    print(end_test)
    
    test_time = end_test-start_test
    print(test_time)
    return f1, train_time, test_time, test_predict_01



def neural_network(split_size):
    logging.getLogger('tensorflow').disabled = True

    random.seed(1)

    trainX_fully_preprocessed, trainY_binary, testX_fully_preprocessed, testY_binary = preprocess_heart_disease_data.Preprocess_Heart_Disease_Data(split_size)
    net=[10, 5]
    F1, train_time, test_time, test_predict = MyDNN(trainX_fully_preprocessed,
                         trainY_binary,
                         testX_fully_preprocessed,
                         testY_binary,
                         hidden_units=net,
                         dropout=0.5,
                         activation_fn=tf.nn.elu,
                         steps=10000)
    print("Results for split percentage: ")
    print("Time taken to train the network: {0}".format(train_time))
    print("Time taken for prediction is :{0}".format(test_time) )
    print("Accuracy of the system using neural network: {0}".format(F1))
    return F1, train_time, test_time


neural_network(0.2)











