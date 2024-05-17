from sklearn.metrics import accuracy_score
import joblib

from common_functions import (
    load_dataset,
    features_target,
    DATA_TYPE,
    PATH_MODEL,
)


def main():

    model = joblib.load(PATH_MODEL)
    data = load_dataset(DATA_TYPE.TEST)
    X, y = features_target(data)

    # Оценка модели
    predictions = model.predict(X)

    print("Метрики при тестировании модели:")
    print(f"accuracy: {accuracy_score(y, predictions)}")


if __name__ == "__main__":
    main()
