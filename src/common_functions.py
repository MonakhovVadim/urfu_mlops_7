from pathlib import Path
from enum import Enum

import joblib
import pandas as pd


NAME_DATASET = "heart_statlog"
DATA_TYPE = Enum("DATA_TYPE", ["BASE", "TRAIN", "TEST"])
PATH_DATASETS = Path.cwd() / "datasets"
PATH_TRAIN = PATH_DATASETS / "train"
PATH_TEST = PATH_DATASETS / "test"
PATH_MODEL = Path.cwd() / "data" / "model.pkl"


def path_by_type(data_type):

    if data_type == DATA_TYPE.BASE:
        path = PATH_DATASETS
    elif data_type == DATA_TYPE.TRAIN:
        path = PATH_TRAIN
    elif data_type == DATA_TYPE.TEST:
        path = PATH_TEST

    return path


def save_dataset(data, data_type, name=NAME_DATASET):

    path = path_by_type(data_type) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path.with_suffix(".csv"), index=False)


def load_dataset(data_type, name=NAME_DATASET):

    path = path_by_type(data_type) / name
    data = pd.read_csv(path.with_suffix(".csv"))
    return data


def features_target(data):
    # Получаем имена предикторов
    features = data.columns.to_list()
    features.remove("target")

    return data[features], data[["target"]]


def save_model(model):

    PATH_MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, PATH_MODEL)
