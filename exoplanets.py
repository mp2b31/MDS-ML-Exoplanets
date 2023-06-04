import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, make_scorer, confusion_matrix
from sklearn.model_selection import GridSearchCV


random_seed = 19950516-19933103
np.random.seed(random_seed)

outlier_ths = {"koi_period":[-np.inf,700],
                "koi_period_err1":[-np.inf,0.06],
                "koi_time0bk":[-np.inf,650],
                "koi_time0bk_err1":[-np.inf,0.2],
                "koi_impact":[-np.inf,20],
                "koi_impact_err1":[-np.inf,12], # Not sure
                "koi_impact_err2":[-10,np.inf],
                "koi_duration":[-np.inf,26],
                "koi_duration_err1":[-np.inf,7],
                "koi_depth":[-np.inf,40000],
                "koi_depth_err1":[-np.inf,600],
                "koi_prad":[-np.inf,200],
                "koi_prad_err1":[-np.inf,50],
                "koi_teq":[-np.inf,4000],
                "koi_insol":[-np.inf,50000],
                "koi_insol_err1":[-np.inf,20000],
                "koi_model_snr":[-np.inf,2200],
                "koi_steff":[-np.inf,8500],
                "koi_steff_err1":[-np.inf,350],
                "koi_slogg":[2,np.inf],
                "koi_slogg_err1":[-np.inf,1],
                "koi_slogg_err2":[-1,np.inf],
                "koi_srad":[-np.inf,30],
                "koi_srad_err1":[-np.inf,4.5],
                "koi_kepmag":[8,np.inf]}


def variable_stats_table(df):
    data_stats = {}
    nrow=len(df)
    for col in df:
        var = df[col]
        data_stats[col] = {}
        data_stats[col]["%miss"] = (var.isnull().sum()/nrow)*100
        data_stats[col]["#unique"] = len(var.drop_duplicates())
        data_stats[col]["type"] = var.dtype.kind
        if data_stats[col]["type"] != "O":
            # Not categorical
            if data_stats[col]["#unique"] > 2:
                # Not Binary
                data_stats[col]["min"] = var.min()
                data_stats[col]["max"] = var.max()
                data_stats[col]["mean"] = var.mean()
                data_stats[col]["median"] = var.median()
                data_stats[col]["std"] = var.std()

    return pd.DataFrame(data_stats).T


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
    float_cols = [ c for c in  df.columns if df[c].dtype.kind == "f"]
    for column in float_cols:
        df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
    
    print("Columns scaled:")
    print(float_cols)
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

def complete_preprocessing(df,normalization=None,alldummy=True):
    """
    Perform preprocessing steps:
    0. Setup target variable
    1. Drop all rows with missing values and unnecessary columns
    2. Convert variables from numerical to categorical
    3. Drop highly correlated variables
    4. Drop outliers
    5. Split data into train and test

    Parameters:
        - df: Raw dataset
        - normalization: Boolean to apply normalization or not
        - alldummy: Boolean to transform all categorical variables to dummy

    Return:
        X_train, X_test, X_val, y_train, y_test, y_val: Three paris of pd.DataFrame pd.Series
            The first element of the pair are always the inpute features and the second the target variable
            The pairs are train, test and validation respectively.
    """

    # Step 0
    df = df.where(df["koi_disposition"] != "FALSE POSITIVE").dropna(axis=0,how="all").set_index("kepoi_name")
    df["koi_disposition"] = df["koi_disposition"].astype("category")

    # Step 1
    exclude_variables = ["kepid","kepler_name","koi_pdisposition","koi_score",
                         "koi_teq_err1","koi_teq_err2","koi_fpflag_co","koi_fpflag_ec","koi_fpflag_nt"]
    df = df.drop(columns=exclude_variables)
    df = df.dropna()

    # Step 2
    df["koi_disposition"] = pd.Categorical(df["koi_disposition"])
    df["koi_tce_plnt_num"] = pd.Categorical(df["koi_tce_plnt_num"],ordered=True)
    df["koi_tce_delivname"] = pd.Categorical(df["koi_tce_delivname"])
    df["koi_fpflag_ss"] = pd.Categorical(df["koi_fpflag_ss"])

    # Step 3
    corr_th = 0.7
    drop_errs = []
    for var in df.columns:
        if var.endswith("err1"):
            var2 = var[:-1]+"2"
            corr = np.abs(np.corrcoef(df[var],df[var2])[1,0])
            if corr > corr_th:
                drop_errs.append(var2)
    df = df.drop(columns=drop_errs)

    # Step 4
    for v in df.columns:
        if df[v].dtype.kind != 'O':
            if v in outlier_ths:
                var = df[v].where((df[v]>outlier_ths[v][0]) & (df[v]<outlier_ths[v][1]))
                df[v] = var
    print("Drop highly correlated error columns:")
    print(drop_errs)
    df = df.dropna()

    # Step 5
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

def plot_confusion_matrix(y_test,y_predict):
    sns.heatmap(confusion_matrix(y_test,y_predict),annot=True,fmt='.0f',cmap="crest",cbar=False,)
    _ = plt.xlabel('Predicted labels')
    _ = plt.ylabel('True labels')
    _ = plt.gca().xaxis.set_ticklabels(['CANDIDATE', 'CONFIRMED'])
    _ = plt.gca().yaxis.set_ticklabels(['CANDIDATE', 'CONFIRMED'])
