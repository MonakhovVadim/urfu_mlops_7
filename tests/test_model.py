from sklearn.metrics import f1_score
import joblib

from common_functions import (
    load_dataset,
    features_target,
    DATA_TYPE,
    PATH_MODEL,
)


def test_model():
    model = joblib.load(PATH_MODEL)
    data = load_dataset(DATA_TYPE.TEST)
    X, y = features_target(data)

    # Оценка модели
    predictions = model.predict(X)
    test_score = f1_score(y, predictions)

    assert test_score > 0.93, f"Failed model test, f1 score on test data: {test_score}"