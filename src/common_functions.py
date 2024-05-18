from pathlib import Path
from enum import Enum

import joblib
import pandas as pd


NAME_DATASET = "heart_statlog"
DATA_TYPE = Enum("DATA_TYPE", ["BASE", "TRAIN", "TEST", "PIPELINE"])
PATH_DATASETS = Path.cwd() / "datasets"
PATH_TRAIN = PATH_DATASETS / "train"
PATH_TEST = PATH_DATASETS / "test"
PATH_MODEL = Path.cwd() / "data" / "model.pkl"
PATH_PIPELINE = Path.cwd() / "ppl" / "ppl.pkl"


def path_by_type(data_type):

    if data_type == DATA_TYPE.BASE:
        path = PATH_DATASETS
    elif data_type == DATA_TYPE.TRAIN:
        path = PATH_TRAIN
    elif data_type == DATA_TYPE.TEST:
        path = PATH_TEST
    elif data_type == DATA_TYPE.PIPELINE:
        path = PATH_PIPELINE

    return path


def save_dataset(data, data_type, name=NAME_DATASET):

    path = path_by_type(data_type) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        data.to_csv(path.with_suffix(".csv"), index=False)
    except PermissionError:
        print(
            "Ошибка доступа! Убедитесь, что у вас есть права на запись в директорию {path}."
        )
    except Exception as e:
        print(f"Ошибка при сохранении датасета {NAME_DATASET}!\n", e)


def load_dataset(data_type, name=NAME_DATASET):

    path = path_by_type(data_type) / name
    try:
        data = pd.read_csv(path.with_suffix(".csv"))
        return data
    except Exception as e:
        print(f"Ошибка при загрузке датасета {NAME_DATASET}!\n", e)
        return None


def save_pipeline(pipeline):

    print(PATH_PIPELINE)
    print(PATH_PIPELINE.parent)

    try:
        PATH_PIPELINE.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, PATH_PIPELINE)
        print("Пайплайн успешно сохранен.")
    except PermissionError:
        print(
            f"Ошибка доступа. Убедитесь, что у вас есть права на запись в директорию {PATH_PIPELINE}."
        )
    except Exception as e:
        print(f"Произошла неизвестная ошибка: {e}")


def load_pipeline():

    try:
        return joblib.load(PATH_PIPELINE)
    except Exception as e:
        print(f"Ошибка при загрузке пайплайна!\n", e)
        return None


def features_target(data):
    # Получаем имена предикторов
    features = data.columns.to_list()
    features.remove("target")

    return data[features], data[["target"]]


def save_model(model):

    PATH_MODEL.parent.mkdir(parents=True, exist_ok=True)
    try:
        joblib.dump(model, PATH_MODEL)
        print("Модель успешно сохранена.")
    except PermissionError:
        print(
            f"Ошибка доступа. Убедитесь, что у вас есть права на запись в директорию {PATH_MODEL}."
        )
    except Exception as e:
        print(f"Произошла неизвестная ошибка: {e}")


def load_model():

    try:
        return joblib.load(PATH_MODEL)
    except Exception as e:
        print(f"Ошибка при загрузке модели!\n", e)
        return None


def desc_dataset():

    bool_features = [
        ("sex", ["female", "male"]),
        ("fasting blood sugar", ["<= 120 mg/dl", "120 mg/dl"]),
        ("exercise angina", ["no", "yes"]),
    ]

    cat_features = [
        (
            "chest pain type",
            ["typical angina", "atypical angina", "non-anginal pain", "asymptomatic"],
        ),
        (
            "resting ecg",
            [
                "normal",
                "having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)",
                "showing probable or definite left ventricular hypertrophy by Estes criteria",
            ],
        ),
        ("ST slope", ["upsloping", "flat", "downsloping"]),
    ]

    num_features = [
        ("age", "in years", 10, 110),
        ("resting bp s", "in mm Hg", 40, 200),
        ("cholesterol", "in mg/dl", 0, 700),
        ("max heart rate", "71–202", 71, 202),
        ("oldpeak", "depression", -5, 7),
    ]

    return {
        "bool_features": bool_features,
        "cat_features": cat_features,
        "num_features": num_features,
    }
