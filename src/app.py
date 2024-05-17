import streamlit as st
import joblib

from common_functions import (
    desc_dataset,
    PATH_MODEL,
)


def main():

    model = joblib.load(PATH_MODEL)
    features = desc_dataset()

    bool_elements = []
    for feature in features["bool_features"]:
        element = st.radio(feature[0], feature[1])
        bool_elements.append(element)

    cat_elements = []
    for feature in features["cat_features"]:
        element = st.radio(feature[0], feature[1])
        cat_elements.append(element)

    num_elements = []
    for feature in features["num_features"]:
        element = st.slider(f"{feature[0]}({feature[0]})", step=1)
        num_elements.append(element)

    if st.button("Определить вероятность болезни"):

        data = []  # Доработать, заменить на данные
        predict = model.predict_proba(data)
        no_seak_score = predict[0][0]
        seak_score = predict[0][1]

        if no_seak_score >= 0.5:
            st.write(f"Вы НЕ больны с вероятностью {no_seak_score}")
        else:
            st.write(f"Вы больны с вероятностью {seak_score}")


if __name__ == "__main__":
    main()
