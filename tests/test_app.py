import pytest
from streamlit.testing.v1 import AppTest

# Исправленная версия теста для Стримлит по основному функционалу приложения

# Импортируем приложение
from app.py import main

@pytest.fixture
def app():
    return AppTest(main)

def test_title(app):
    app.run()
    assert app.title[0].value == "Определение риска сердечно-сосудистого заболевания :sparkling_heart:"

def test_button_click(app):
    app.run()
    # Предполагаем, что пользователь выбрал определенные значения для кнопок и слайдеров
    user_choice = {
        'bool_feature_1': 1, # Значение, возвращаемое первой кнопкой
        'cat_feature_1': 2,  # Значение, возвращаемое второй кнопкой
        'num_feature_1': 50  # Значение, установленное пользователем на слайдере
    }
    # Устанавливаем значения виджетов в соответствии с выбором пользователя
    for feature, value in user_choice.items():
        app.set_widget_value(feature, value)
    # Нажимаем кнопку для предсказания
    app.button("Определить вероятность болезни").click().run()
    # Проверяем вывод после нажатия кнопки
    assert "Вы НЕ больны с вероятностью" in app.markdown[0].value or \
           "Вы больны с вероятностью" in app.markdown[0].value
