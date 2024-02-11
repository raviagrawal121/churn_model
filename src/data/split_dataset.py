from sklearn.model_selection import train_test_split

def split_data(df,train_data_path,test_data_path,split_ratio,random_state):
    train, test = train_test_split(df, test_size=split_ratio, random_state=random_state)
    train.to_csv(train_data_path, sep=",", index=False, encoding="utf-8")
    test.to_csv(test_data_path, sep=",", index=False, encoding="utf-8")   