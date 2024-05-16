from sklearn.ensemble import RandomForestClassifier
import joblib

from common_functions import (
    features_target,
    load_dataset,
    save_model,
    NAME_DATASET,
    SUF_PREPROCESSING,
)


def main():
    # Загружаем датасет
    data = load_dataset(NAME_DATASET + SUF_PREPROCESSING)

    # Получаем имена предикторов и целевого признака
    X, y = features_target(data)

    # Обучаем модель
    model = RandomForestClassifier()
    model.fit(X, y)

    # Сохраняем модель
    save_model(model)


if __name__ == "__main__":
    main()
