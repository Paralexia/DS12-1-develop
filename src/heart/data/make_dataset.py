# -*- coding: utf-8 -*-
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def read_data(dataset_path: str) -> pd.DataFrame:
    """Reading dataset from path"""
    data = pd.read_csv(dataset_path)
    return data


def split_train_test_data():
    """Split dataset into random train and test subsets"""
    # W.I.P
    pass
