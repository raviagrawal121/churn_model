# -*- coding: utf-8 -*-
import argparse
from src import logger
from pathlib import Path
from load_dataset import load_data
from split_dataset import split_data
from src.utils.common import read_params

def main(config_path):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger.info('making final data set from raw data')

    config = read_params(config_path)
    external_data_path = config["external_data_config"]["external_data_csv"]
    raw_data_path = config["raw_data_config"]["raw_data_csv"]
    model_var = config["raw_data_config"]["model_var"]
    train_data_path = config["processed_data_config"]["train_data_csv"]
    test_data_path = config["processed_data_config"]["test_data_csv"]
    split_ratio = config["raw_data_config"]["train_test_split_ratio"]
    random_state = config["raw_data_config"]["random_state"]

    df = load_data(external_data_path, model_var)
    split_data(df, train_data_path, test_data_path, split_ratio, random_state)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()
    main(config_path=args.config)
