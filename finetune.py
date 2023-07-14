import argparse
import yaml
import pandas as pd
from finetuner import Finetuner

def load_yaml(path:str) -> dict:
    return yaml.full_load(open(path, 'r'))

def load_df(path: str) -> pd.DataFrame:
    return pd.read_pickle(path)

def main(config_path:str = 'config/training_config.yml', data_path:str = 'data/data.pkl') -> None:
    '''
    We load the config and data to perform finetuning
    '''
    config = load_yaml(config_path)
    data = load_df(data_path)
    finetuner = Finetuner(data, **config)
    finetuner.train()
    finetuner.evaluate()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "Fine Tuning Bert model on task's data")
    parser.add_argument('-c', '--config', help='Training configuration', default='config/training_config.yml')
    parser.add_argument('-d', '--data', help='Our pre-processed dataset', default='data/data.pkl')
    args = parser.parse_args()

    main(args.config, args.data)