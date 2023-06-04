import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, make_scorer
from keras.utils.vis_utils import plot_model

from exoplanets import *


random_seed = 19950516-19933103
np.random.seed(random_seed)


def build_binary_classification_model(input_shape):
    """
    Function to build a neural network model for binary classification.
    
    Parameters:
        - input_shape: The shape of the input features.
        
    Returns:
        - The built neural network model.
    """
    model = Sequential()
    
    # Add input layer
    model.add(Dense(64, activation='tanh', input_shape=input_shape))
    
    # Add hidden layers
    model.add(Dense(84, activation='tanh'))
    model.add(Dense(128, activation='tanh'))
    model.add(Dense(2, activation='tanh'))
    model.add(Dense(64, activation='tanh'))
    
    # Add output layer
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model


### --- MAIN --- ###
df = pd.read_csv('exoplanets.csv')
X_train, X_test, X_val, y_train, y_test, y_val = complete_preprocessing(df,normalization="Min-Max",alldummy=True)

input_shape = (X_train.shape[1],)

# NN adhoc preprocessing:
X = X_train.astype(float).values
y = y_train.cat.codes.values
X_test = X_test.astype(float).values
y_test = y_test.cat.codes.values
X_val = X_val.astype(float).values
y_val = y_val.cat.codes.values

model = build_binary_classification_model(input_shape)
history = model.fit(X, y, epochs=50, batch_size=36, validation_data=(X_test,y_test))

# Plot NN architecture
#plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True, show_layer_activations=True)

predictions = model.predict(X_test).round(0)
#print(classification_report(y_test, predictions))
#plot_confusion_matrix(y_test,predictions)
#plt.show()

## Plot accuracy evolution
#train_acc = history.history['accuracy']
#val_acc = history.history['val_accuracy']
#plt.plot(train_acc, label='Training Accuracy')
#plt.plot(val_acc, label='Validation Accuracy')
#plt.xlabel('Epochs')
#plt.ylabel('Accuracy')
#plt.legend()
#plt.savefig("plot.png")
#plt.show()

# Validation
print("Testing model in validation dataset")
predictions = model.predict(X_val).round(0)
print(classification_report(y_val, predictions))
plot_confusion_matrix(y_val,predictions)
plt.show()


