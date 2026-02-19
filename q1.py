import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

def preprocessing():

    #part a
    (x_train_full, y_train_full), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    print("Original Data Shapes:")
    print(f"Train: {x_train_full.shape}, Labels: {y_train_full.shape}")
    print(f"Test:  {x_test.shape}, Labels: {y_test.shape}")
    print("-" * 30)

    #part b
    x_train_full = x_train_full.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0

    #part c
    num_classe = 10
    y_train_full_encoded = tf.keras.utils.to_categorical(y_train_full, num_classe)
    y_test_encoded = tf.keras.utils.to_categorical(y_test, num_classe)

    #part d
    x_train,x_val,y_train,y_val = train_test_split(x_train_full,y_train_full_encoded,test_size=0.2,random_state=42)
    print("FINAL DATA SHAPES (Ready for Model):")
    print(f"Training Set:   {x_train.shape}")
    print(f"Validation Set: {x_val.shape}")
    print(f"Test Set:       {x_test.shape}")

    return x_train,x_val,x_test,y_train,y_val,y_test,y_train_full_encoded,y_test_encoded
