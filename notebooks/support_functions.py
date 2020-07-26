from sklearn.model_selection import train_test_split
import sklearn
from sklearn.metrics import plot_confusion_matrix

def split_data(X, y):
    """
    Shows train and test error
    Parameters
    ----------
    model: sklearn classifier model
        The sklearn model
    X: numpy.ndarray
        The X part (features) of the dataset
    y numpy.ndarray
        The y part (target) of the dataset
    Returns
    -------
        X_train: numpy.ndarray
            The X part of the train dataset
        y_train: numpy.ndarray
            The y part of the train dataset
        X_valid: numpy.ndarray
            The X part of the validation dataset
        y_valid
            The y part of the validation dataset
        X_trainvalid
            The X part of the train+validation dataset
        y_trainvalid
            The y part of the train+validation dataset
        X_test
            The X part of the test dataset
        y_test
            The y part of the test dataset
    """
    X_trainvalid, X_test, y_trainvalid, y_test = train_test_split(X, y, train_size=0.8, random_state=22)
    X_train, X_valid, y_train, y_valid = train_test_split(X_trainvalid, y_trainvalid,
                                                          train_size=0.75, random_state=22)

    print("Number of training examples:", len(y_train))
    print("Number of validation examples:", len(y_valid))
    print("Number of test examples:", len(y_test))

    return X_train, y_train, X_valid, y_valid, X_trainvalid, y_trainvalid, X_test, y_test


def get_scores(model, 
                X_train, y_train,
                X_valid, y_valid, 
                show = True
               ):
    """
    Returns train and validation error given a model
    train and validation X and y portions
    Parameters
    ----------
    model: sklearn classifier model
        The sklearn model
    X_train: numpy.ndarray        
        The X part of the train set
    y_train: numpy.ndarray
        The y part of the train set    
    X_valid: numpy.ndarray        
        The X part of the validation set
    y_valid: numpy.ndarray
        The y part of the validation set    
    Returns
    -------
        train_err: float
        test_err: float
            
    """    
    if show: 
        print("Training error:   %.2f" % (1-model.score(X_train, y_train)))
        print("Validation error: %.2f" % (1-model.score(X_valid, y_valid)))
        print('\n')
    return (1-model.score(X_train, y_train)), (1-model.score(X_valid, y_valid))

def get_scores_reg(model, 
                X_train, y_train,
                X_valid, y_valid, 
                show = True
               ):
    """
    Returns train and validation error given a model
    train and validation X and y portions
    Parameters
    ----------
    model: sklearn classifier model
        The sklearn model
    X_train: numpy.ndarray        
        The X part of the train set
    y_train: numpy.ndarray
        The y part of the train set    
    X_valid: numpy.ndarray        
        The X part of the validation set
    y_valid: numpy.ndarray
        The y part of the validation set    
    Returns
    -------
        train_err: float
        test_err: float
            
    """    
    if show: 
        print("Training error:   %.2f" % (model.score(X_train, y_train)))
        print("Validation error: %.2f" % (model.score(X_valid, y_valid)))
        print('\n')
    return (model.score(X_train, y_train)), (model.score(X_valid, y_valid))

def display_confusion_matrix_classification_report(model, X_valid, y_valid, 
                                                   labels=['Non fraud', 'Fraud'],
                                                   confusion_matrix=True):
    """
    Displays confusion matrix and classification report. 
    
    Arguments
    ---------     
    model -- sklearn classifier model
        The sklearn model
    X_valid -- numpy.ndarray        
        The X part of the validation set
    y_valid -- numpy.ndarray
        The y part of the validation set       
    Keyword arguments:
    -----------
    labels -- list (default = ['Non fraud', 'Fraud'])
        The labels shown in the confusion matrix
    Returns
    -------
        None
    """
    if confusion_matrix: 
        ### Display confusion matrix 
        disp = plot_confusion_matrix(model, X_valid, y_valid,
                                     display_labels=labels,
                                     cmap=plt.cm.Blues, 
                                     values_format = 'd')
        disp.ax_.set_title('Confusion matrix for the dataset')

    ### Print classification report
    print(classification_report(y_valid, model.predict(X_valid)))