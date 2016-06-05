import seaborn as sns
import numpy as np
import pandas as pd

from sklearn import neighbors, svm
from sklearn.cross_validation import train_test_split


def split(df, test_column, test_size, random_state):
    """
    Uses sklearn.train_test_split to split "df" into a testing set and a test set.
    The "test_columns" lists the column that we are trying to predict.
    All columns in "df" except "test_columns" will be used for training.
    The "test_size" should be between 0.0 and 1.0 and represents the proportion of the
    dataset to include in the test split.
    The "random_state" parameter is used in sklearn.train_test_split.

    Parameters
    ----------
    df: A pandas.DataFrame
    test_column: A string
    test_size: A float
    random_state: A numpy.random.RandomState instance

    Returns
    -------
    A 4-tuple of pandas.DataFrames
    """

    # choose only non test-column for training data by calling df.ix
    return train_test_split(df.ix[:, df.columns != test_column], df[test_column],
                            test_size=test_size, random_state=random_state)


def normalize(df):
    """
    Takes a dataframe and normlizes features to be in range [0, 1].

    Parameters
    ----------
    df: A pandas.DataFrame

    Returns
    -------
    A pandas.DataFrame
    """
    result = (df - df.min()) / (df.max() - df.min())  # just use the formula above to normalize

    return result


def train_knn(X, y, n_neighbors):
    """
    Fits a $k$-Nearest Neighbors on the training data.
    Returns the trained model (an `sklearn.neighbors.KNeighborsClassifier` object).

    Parameters
    ----------
    X: A pandas.DataFrame. Training attributes.
    y: A pandas.DataFrame. Truth labels.
    n_neighbors: Number of neighbors to use in kNN.

    Returns
    -------
    An sklearn.neighbors.KNeighborsClassifier object.
    """
    knc = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors)
    model = knc.fit(X, y.values.ravel())

    return model


def predict_knn(model, X):
    """
    Fits an `sklearn.neighbors.KNeighborsClassifier` model on `X` and
    returns a `numpy.ndarray` of predicted values.

    Parameters
    ----------
    model: An sklearn.neighbors.KNeighborsClassifier object.
    X: pandas.DataFrame

    Returns
    -------
    A pandas.DataFrame. Has one column "Delayed".
    """

    prediction = model.predict(X)

    # Create a dictionary with our key as 'Delayed' and our value as the array returned from knc.predict()
    d = {'Delayed': prediction}

    # We need to turn the Dict into a Pandas DataFrame with one column
    df = pd.DataFrame(d)

    return df


def compute_accuracy(y_train,y_test, column):
    """
    Compute accuracy between two data frames of the same size


    Parameters
    ----------
    y_train: A pandas.DataFrame
    y_test: A pandas.DataFrame
    column: A string

    Returns
    -------
    A float
    """

    if y_train.shape[0] != y_test.shape[0]:
        print('The DataFrames are not of the same length')
        return 0

    score = pd.Series.mean(y_train[column]==y_test[column])

    return score


def plot_confusion(y_pred, y_test, text):
    """
    Plots a confusion matrix using numpy.histogram2d() and seaborn.heatmap().
    Returns a maptlotlib.axes.Axes instance.
    """
    test = np.ravel(np.array(y_test))
    pred = np.ravel(np.array(y_pred))
    pts, xe, ye = np.histogram2d(test, pred, bins=2)
    pd_pts = pd.DataFrame(pts.astype(int), index=['Not delayed', 'Delayed'], columns=['Not delayed', 'Delayed'])
    sns.set(font_scale=1.5)
    ax = sns.heatmap(pd_pts, annot=True, fmt="d")

    sns.set(font_scale=2.0)
    ax.axes.set_title("Confusion Matrix for "+text)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

    return ax


def get_score(y_train,y_test, column):
    """
    Compute score between two data frames of the same size


    Parameters
    ----------
    y_train: A pandas.DataFrame
    y_test: A pandas.DataFrame
    column: A string

    Returns
    -------
    An int
    """

    x = list(y_train[column])
    y = list(y_test[column])

    if len(x) != len(y):
        print('The DataFrames are not of the same length')
        return 0

    score = 0

    for i in range(len(x)):
        if (x[i] == 0) & (y[i] == 0):
            score += 0

        if (x[i] == 0) & (y[i] == 1):
            score -= -8

        if (x[i] == 1) & (y[i] == 1):
            score += +8

        if (x[i] == 1) & (y[i] == 0):
            score -= 0

    return score


def standardize(x):
    """
    Takes a 2d array and normlizes each feature (each column) to be in range [-1, 1].

    Parameters
    ----------
    array: A numpy.ndarray

    Returns
    -------
    A numpy.ndarray
    """

    result = list()
    x = np.array(x)

    for col in x.transpose():
        r = [((a - col.min()) / (col.max() - col.min()) - .5) * 2 for a in col]
        result.append(r)

    scaled = np.array(result).transpose()

    return scaled


def fit_and_predict(X_train, y_train, X_test, kernel):
    """
    Fits a Support Vector Machine on the training data on "X_train" and "y_train".
    Returns the predicted values on "X_test".

    Parameters
    ----------
    X_train: A numpy.ndarray
    y_train: A numpy.ndarray
    X_test: A numpy.ndarray
    kernel: A string that specifies kernel to be used in SVM

    Returns
    -------
    model: An svm.SVC instance trained on "X_train" and "y_train"
    y_pred: A numpy array. Values predicted by "model" on "X_test"
    """

    model = svm.SVC(kernel=kernel, C=1)
    model.fit(X_train, y_train.values.ravel())

    y_pred = model.predict(X_test)

    return model, y_pred


def separate_by_class(X, y):
    """
    Separate the training set ("X") by class value ("y")
    so that we can calculate statistics for each class.

    Parameters
    ----------
    X: A 2d numpy array
    y: A 1d numpy array
    Returns
    -------
    A dictionary of 2d numpy arrays
    """

    separated = dict()

    for i in range(len(y)):

        if y[i] not in separated.keys():
            separated[y[i]] = [X[i]]
        else:
            separated[y[i]].append(X[i])

    return separated


def summarize(X):
    """
    For a given list of instances (for a class value),
    calculates the mean and the standard deviation for each attribute.

    Parameters
    ----------
    A 2d numpy array

    Returns
    -------
    A 2d numpy array
    """

    return np.array([(np.mean(x, axis=0), np.std(x,axis=0,ddof=1)) for x in X.transpose()])


def summarize_by_class(X, y):
    """
    Separates a training set into instances grouped by class.
    It then calculates the summaries for each attribute.

    Parameters
    ----------
    X: A 2d numpy array. Represents training attributes.
    y: A 1d numpy array. Represents class labels.
    Returns
    -------
    A dictionary of 2d numpy arrays
    """

    separated = separate_by_class(X, y)
    summaries = dict()

    for k in separated.keys():
        summaries[k] = summarize(np.array(separated[k]))

    return summaries


def calculate_log_probability(x, mean, stdev):
    """
    Calculates log of Gaussian function to estimate
    the log probability of a given attribute value.

    Parameters
    ----------
    x: A float or 1d numpy array
    mean: A float or 1d numpy array
    stdev: A float or 1d numpy array

    Returns
    -------
    A float or 1d numpy array
    """

    exponent = np.exp(-(np.power(x - mean, 2) / (2 * np.power(stdev, 2))))

    return np.log((1 / (np.sqrt(2 * np.pi) * stdev)) * exponent)


def calculate_class_log_probabilities(summaries, input_array):
    """
    Combines the probabilities of all of the attribute values for a data instance
    and comes up with a probability of the entire data instance belonging to the class.

    Parameters
    ----------
    summaries: A dictionary of 2d numpy arrays
    input_array: A 1d numpy array

    Returns
    -------
    A dictionary of log probabilities
    """

    log_probabilities = dict()

    for k in summaries.keys():
        log_probabilities[k] = 0

        for s in summaries[k]:
            p = calculate_log_probability(input_array, s[0], s[1])
            log_probabilities[k] += p

    return log_probabilities


def predict(summaries, input_array):
    """
    Calculates the probability of each data instance belonging to each class value,
    looks for the largest probability, and return the associated class.

    Parameters
    ----------
    summaries: A dictionary of numpy arrays
    input_array: A 1d numpy array

    Returns
    -------
    A 1d numpy array
    """

    p = calculate_class_log_probabilities(summaries, input_array)

    return np.greater(p[1], p[0])