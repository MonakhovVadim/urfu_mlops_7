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

    for i in range(10):

        test = X.iloc[[i]]
        fact = y.iloc[[i]]["target"].values[0]

        print(f"Тестируем запись с индексом {i}. Фактическая цель: {fact}")

        predict = model.predict_proba(test)
        no_seak_score = predict[0][0]
        seak_score = predict[0][1]

        print("Результат модели:")
        if no_seak_score >= 0.5:
            print(f"Вы НЕ больны с вероятностью {no_seak_score}")
        else:
            print(f"Вы больны с вероятностью {seak_score}")
        print("")


if __name__ == "__main__":
    main()
