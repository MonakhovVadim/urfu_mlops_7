import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from common_functions import load_dataset, save_dataset, features_target, DATA_TYPE


def main():

    # Загружаем датасет
    data = load_dataset(DATA_TYPE.BASE)
    X, y = features_target(data)

    # Создаем пайплайн с предобработкой
    pipeline = Pipeline([("scaler", StandardScaler())])

    # Делаем преобразования
    data = pd.DataFrame(pipeline.fit_transform(X), columns=X.columns)
    data[y.columns] = y

    # Делим на тренировочные и тестовые данные
    data_train, data_test = train_test_split(data, test_size=0.2, random_state=42)

    # Сохраняем датасет для тренировки
    save_dataset(data_train, DATA_TYPE.TRAIN)

    # Сохраняем датасет для тестирования
    save_dataset(data_test, DATA_TYPE.TEST)


if __name__ == "__main__":
    main()
