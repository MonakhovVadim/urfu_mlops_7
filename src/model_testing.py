from sklearn.metrics import accuracy_score
import joblib

from common_functions import (
    load_dataset,
    features_target,
    NAME_DATASET,
    SUF_PREPROCESSING,
    PATH_MODEL,
)


def main():

    model = joblib.load(PATH_MODEL)
    data = load_dataset(NAME_DATASET + SUF_PREPROCESSING)
    X, y = features_target(data)

    # Оценка модели
    predictions = model.predict(X)

    print("Метрики при тестировании модели:")
    print(f"accuracy: {accuracy_score(y, predictions)}")


if __name__ == "__main__":
    main()
