import streamlit as st
import joblib
import pandas as pd
from common_functions import (
    load_model,
    load_pipeline,
)


from common_functions import (
    desc_dataset,
    PATH_MODEL,
)


def main():

    # загружаем модель и пайплайн для скелера введенных пользователем данных
    model = load_model()
    pipeline = load_pipeline()

    # выводим приверственный тайтл и кратко обозначаем, что делает помощник
    st.title("Определение риска сердечно-сосудистого заболевания :sparkling_heart:")
    st.write(
        """Настоящая модель осуществляет предсказание сердечно-сосудистого заболеия 
        пациента на основании введеных пользователем параметров. 
        Модель обучена на датасете Heart Disease UCI. 
        """
    )

    features = desc_dataset()

    # объединяем при выводе булевы признаки с категориальными
    # поскольку в любом случае будем использоать единый визуаьный интерфейс
    radio_elements = []
    bfel = features["bool_features"] + features["cat_features"]
    for feature in bfel:
        element = st.radio(*feature)
        radio_elements.append(element)

    num_elements = []
    for feature in features["num_features"]:
        element = st.slider(
            f"{feature[0]} ({feature[1]})", min_value=feature[2], max_value=feature[3]
        )
        num_elements.append(element)

    if st.button("Определить вероятность болезни"):
        user_choice = {}
        # поскольку st.radio возвращает текст кнопки, нужно перевести их обратно в числа
        # и заодно формируем словарь с выбором пользователя
        for feature, rel in zip(bfel, radio_elements):
            user_choice[feature[0]] = feature[1].index(rel)

        for feature, nel in zip(features["num_features"], num_elements):
            user_choice[feature[0]] = nel

        # переводим пользовательский вывод в датафрейм
        df = pd.DataFrame(user_choice, index=[0])

        print("Датафрейм с введенным пользоваталем данными\n", df)

        # приводим столбцы в правильное расположение (как в изначальном датафрейме)
        df = df[pipeline.feature_names_in_]

        print("Датафрейм с корректной последовательность колонок\n", df)
        # преобразуем сырые данные по пайплайну, который использовался при преобразовании датасета
        df_scaled = pipeline.transform(df)
        print("Преобразованный датафрейм\n", df_scaled)

        predict = model.predict_proba(pd.DataFrame(df_scaled, columns=df.columns))
        no_seak_score = predict[0][0]
        seak_score = predict[0][1]

        if no_seak_score >= 0.5:
            st.write(f"Вы НЕ больны с вероятностью {no_seak_score}")
        else:
            st.write(f"Вы больны с вероятностью {seak_score}")


if __name__ == "__main__":
    main()
