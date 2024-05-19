import pandas as pd
import pytest

from common_functions import load_dataset, DATA_TYPE


def test_dataset_structure():
    # Проверяем, что функция загрузки датасета работает корректно
    data = load_dataset(DATA_TYPE.BASE, "heart_statlog")
    assert isinstance(data, pd.DataFrame), "Loaded data is not a DataFrame"
    # Проверяем структуру датасета
    expected_columns = ['age', 'sex', 'chest pain type', 'resting bp s', 'cholesterol',
                        'fasting blood sugar', 'resting ecg', 'max heart rate',
                        'exercise angina', 'oldpeak', 'ST slope', 'target']
    assert list(data.columns) == expected_columns, "Dataset columns do not match expected columns"


def test_dataset_values():
    # Проверяем, что датасет не содержит недопустимых значений
    data = load_dataset(DATA_TYPE.BASE, "heart_statlog")
    assert data.isnull().sum().sum() == 0, "Dataset contains null values"
    assert not data.isin(
        [float('inf'), float('-inf')]).any().any(), "Dataset contains infinite values"


def test_missing_values():
    limit_percentage = 5
    # Проверяем, что датасет содержит менее limit_percentage процентов пропущенных значений
    data = load_dataset(DATA_TYPE.BASE, "heart_statlog")
    missing_values = data.isnull().mean() * 100  # процент пропущенных значений по каждому столбцу
    assert not (
            missing_values > limit_percentage).any(), \
        (f"Dataset contains columns with more than {limit_percentage}% "
         f"missing values: {missing_values[missing_values > limit_percentage]}")


def test_negative_values():
    # Проверяем, что датасет не содержит отрицательных значений там, где они не допустимы
    data = load_dataset(DATA_TYPE.BASE, "heart_statlog")
    columns_to_check = ['age', 'cholesterol', 'resting bp s', 'max heart rate']
    for column in columns_to_check:
        assert (data[column] >= 0).all(), \
            f"Dataset contains negative values in column {column}"


def test_anomalous_values():
    # Проверяем, что датасет не содержит аномальных значений
    data = load_dataset(DATA_TYPE.BASE, "heart_statlog")
    assert data['max heart rate'].between(60, 202, "both").all(),\
        "Dataset contains anomalous values in 'max heart rate' column"

def test_data_types():
    # Проверяем, что типы данных в датасете соответствуют ожидаемым
    data = load_dataset(DATA_TYPE.BASE, "heart_statlog")
    assert data['age'].dtype == 'int64', "Incorrect data type for 'age'"
    assert data['resting bp s'].dtype == 'int64', "Incorrect data type for 'resting bp s'"
    assert data['cholesterol'].dtype == 'int64', "Incorrect data type for 'cholesterol'"
    assert data['max heart rate'].dtype == 'int64', "Incorrect data type for 'max heart rate'"
    assert data['oldpeak'].dtype == 'float64', "Incorrect data type for 'oldpeak'"
    for column in ['sex', 'fasting blood sugar', 'exercise angina', 'target']:
        assert data[column].dtype == 'int64', f"Incorrect data type for '{column}'"
        assert set(data[column].unique()).issubset({0, 1}), f"Invalid values in '{column}'"
