from pathlib import Path

import joblib
import pandas as pd

PATH_DATASETS = Path.cwd() / "datasets"
NAME_DATASET = "heart_statlog"
SUF_PREPROCESSING = "_preprocessing"

PATH_MODEL = Path.cwd() / "data" / "model.pkl"


def save_dataset(data, name):
    way = PATH_DATASETS / name
    way.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(way.with_suffix(".csv"), index=False)


def load_dataset(name):
    way = PATH_DATASETS / name
    data = pd.read_csv(way.with_suffix(".csv"))
    return data


def features_target(data):
    # Получаем имена предикторов
    features = data.columns.to_list()
    features.remove("target")

    return data[features], data[["target"]]


def save_model(model):

    PATH_MODEL.parent.mkdir(parents=True, exist_ok=True)
    # Сохраняем модель
    joblib.dump(model, PATH_MODEL)
