import yaml
from sklearn.metrics import f1_score,recall_score,accuracy_score,precision_score,confusion_matrix,classification_report
from src import logger

def read_params(config_path):
    """
    read parameters from the params.yaml file
    input: params.yaml location
    output: parameters as dictionary
    """
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def accuracymeasures(y_test,predictions,avg_method):
    """
    calculate accuracy,precision,recall,f1score
    input: y_test,predictions,avg_method
    output: accuracy,precision,recall,f1score
    """

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average=avg_method)
    recall = recall_score(y_test, predictions, average=avg_method)
    f1score = f1_score(y_test, predictions, average=avg_method)
    target_names = ['0','1']

    print("Classification report")
    print("---------------------","\n")
    classification_rep = classification_report(y_test, predictions, target_names=target_names)
    print(classification_rep,"\n")
    logger.info("Classification report:\n%s", classification_rep)

    print("Confusion Matrix")
    print("---------------------","\n")
    confusion_mat = confusion_matrix(y_test, predictions)
    print(confusion_mat,"\n")
    logger.info("Confusion Matrix:\n%s", confusion_mat)

    print("Accuracy Measures")
    print("---------------------","\n")
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 Score: ", f1score)
    logger.info("Accuracy Measures:")
    logger.info("---------------------")
    logger.info("Accuracy: %s", accuracy)
    logger.info("Precision: %s", precision)
    logger.info("Recall: %s", recall)
    logger.info("F1 Score: %s", f1score)

    return accuracy,precision,recall,f1score

def get_feat_and_target(df,target):
    """
    Get features and target variables seperately from given dataframe and target 
    input: dataframe and target column
    output: two dataframes for x and y 
    """
    logger.info("Extracting features and target variables...")

    x=df.drop(target,axis=1)
    y=df[[target]]

    logger.info("Features and target variables extracted successfully.")
    return x,y 