from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, make_scorer


random_seed = 19950516-19933103
np.random.seed(random_seed)

def fine_tune_model(model, param_grid, X, y, cv=5):
    """
    Function for fine-tuning a machine learning model using cross-validation.
    
    Parameters:
        - model: The model object to be fine-tuned.
        - param_grid: A dictionary or a list of dictionaries containing the parameter grid.
        - X: The input features.
        - y: The target variable.
        - cv: The number of cross-validation folds (default: 5).
        
    Returns:
        - The best model obtained after fine-tuning.
    """
    # Define f1 metric 
    f1 = make_scorer(f1_score , average='macro')
    
    # Create a GridSearchCV object
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, scoring=f1)

    # Fit the GridSearchCV object to the data
    grid_search.fit(X, y)

    # Print the best parameters found
    print("Best Parameters:", grid_search.best_params_)
    
    # Return the best model obtained
    return grid_search.best_estimator_

def convert_to_dummies(df):
    # Get columns of type "object" or "category"
    object_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Convert columns to dummy variables
    df_dummies = pd.get_dummies(df, columns=object_cols)

    print("Columns transformed to dummy:")
    print(object_cols)

    return df_dummies

def min_max_scaling(df):
    # Perform Min-Max scaling
    for column in df.columns:
        if df[column].dtype.kind == "f":
            print(f"Scaling {column}")
            df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
    return df


def preprocessing(df,normalization=None,alldummy=True):
    """
    Perform preprocessing steps:
    1. Drop all rows with missing values
    2. Convert variables from numerical to categorical
    3. Drop highly correlated variables
    4. Drop outliers
    5. Split data into train and test
    """

    y = df['koi_disposition'].copy()
    X = df.drop('koi_disposition', axis=1).copy()

    if alldummy:
        X = convert_to_dummies(X)

    if normalization == "Min-Max":
        X = min_max_scaling(X)

    # Validation split (10%)
    X, X_val, y, y_val = train_test_split(X, y, test_size=0.1, random_state=101)

    # Train test split (70% - 30%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


    return X_train, X_test, X_val, y_train, y_test, y_val

# https://stackoverflow.com/questions/45411902/how-to-use-f1-score-with-keras-model
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
df = pd.read_csv("saved.df.csv",index_col=0)
df["koi_disposition"] = pd.Categorical(df["koi_disposition"])

X_train, X_test, X_val, y_train, y_test, y_val = preprocessing(df,normalization="Min-Max",alldummy=True)

input_shape = (X_train.shape[1],)

X = X_train.astype(float).values
y = y_train.cat.codes.values

#model = KerasClassifier(model=build_binary_classification_model,epoch=100, batch_size=5, verbose=1)
model = build_binary_classification_model(input_shape)
model.fit(X, y, epochs=50, batch_size=36)

predictions = model.predict(X_test.astype(float).values).round(0)
print(classification_report(y_test.cat.codes.values, predictions))

