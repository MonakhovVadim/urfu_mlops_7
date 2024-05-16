import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from common_functions import (
    load_dataset,
    save_dataset,
    features_target,
    NAME_DATASET,
    SUF_PREPROCESSING,
)


def main():

    # Загружаем датасет
    data = load_dataset(NAME_DATASET)
    X, y = features_target(data)

    # Создаем пайплайн с предобработкой
    pipeline = Pipeline([("scaler", StandardScaler())])

    # Делаем преобразования
    data = pd.DataFrame(pipeline.fit_transform(X), columns=X.columns)
    data[y.columns] = y

    # Сохраняем датасет
    save_dataset(data, NAME_DATASET + SUF_PREPROCESSING)


if __name__ == "__main__":
    main()
