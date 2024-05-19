import streamlit as st
from streamlit.testing.v1 import AppTest

app_test = AppTest.from_file ("test_app.py")

# Тестирование заголовка приложения
def test_title():
    app_test.run()
    assert 'Ожидаемый Заголовок' in app_test.get(st.title)

# Тестирование наличия определенного текста
def test_text():
    app_test.run()
    assert 'Ожидаемый текст' in app_test.get(st.text)

# Тестирование наличия кнопки
def test_button():
    app_test.run()
    assert app_test.get(st.button, label='Ожидаемая метка кнопки')

# Тестирование значения в селектбоксе
def test_selectbox():
    app_test.run()
    assert app_test.get(st.selectbox, label='Ожидаемая метка селектбокса').options == ['Опция 1', 'Опция 2']

# Тестирование состояния сессии
def test_session_state():
    app_test.run()
    assert app_test.session_state['ключ_состояния'] == 'Ожидаемое значение'
