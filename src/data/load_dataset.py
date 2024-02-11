import pandas as pd 

def load_data(data_path,model_var):
    """
    load csv dataset from given path
    input: csv path 
    output:pandas dataframe 
    note: Only 6 variables are used in this model building stage for the simplicity.
    """
    df = pd.read_csv(data_path, sep=",", encoding='utf-8')
    df=df[model_var]
    return df
    