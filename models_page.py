# models_page.py
import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Функция для загрузки данных и предварительной обработки
def load_data():
    # Чтение данных
    med_df = pd.read_csv('Datasets\data_cleaned.csv')
    
    # Подготовка признаков
    X = med_df[['cleaned_symptoms', 'cleaned_anamnesis']].fillna('')
    y = med_df['icd10']
    
    # Векторизация текста
    tfidf_vectorizer = TfidfVectorizer()
    X_vectorized = tfidf_vectorizer.fit_transform(X['cleaned_symptoms'].fillna('') + ' ' + X['cleaned_anamnesis'].fillna(''))

    # Стандартизация данных
    scaler = StandardScaler(with_mean=False)  # with_mean=False для разреженных матриц
    X_scaled = scaler.fit_transform(X_vectorized)

    # Разделение данных на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Применение PCA
    pca = PCA(n_components=200)
    X_train_reduced = pca.fit_transform(X_train)
    X_test_reduced = pca.transform(X_test)

    # Использование SelectKBest для отбора признаков
    selector = SelectKBest(f_classif, k=100)
    X_train_selected = selector.fit_transform(X_train_reduced, y_train)
    X_test_selected = selector.transform(X_test_reduced)

    return X_train_selected, X_test_selected, y_train, y_test, tfidf_vectorizer, pca, selector

# Функция для предсказания диагноза
def predict_diagnosis(model, input_data, tfidf_vectorizer, pca, selector, mkb_df):
    # Векторизация и уменьшение размерности
    new_data_vectorized = tfidf_vectorizer.transform(input_data)
    new_data_reduced = pca.transform(new_data_vectorized.toarray())
    new_data_selected = selector.transform(new_data_reduced)
    
    # Предсказание модели
    prediction = model.predict(new_data_selected)
    
    # Получаем диагноз по коду
    diagnosis_code = prediction[0]
    diagnosis_name = mkb_df.loc[mkb_df['Код'] == diagnosis_code, 'Название'].values[0]
    
    return f"Предсказанный диагноз: {diagnosis_code} - {diagnosis_name}"

# Страница с моделями
def show_models_page():
    st.title("Модели для анализа данных")

    # Загружаем данные и обучаем модели
    X_train_selected, X_test_selected, y_train, y_test, tfidf_vectorizer, pca, selector = load_data()
    
    # Загрузка обученных моделей
    with open("Models/sgd_model.pkl", "rb") as f:
        sgd_model = pickle.load(f)
    
    with open("Models/dt_model.pkl", "rb") as f:
        dt_model = pickle.load(f)
    
    with open("Models/rf_model.pkl", "rb") as f:
        rf_model = pickle.load(f)
    
    # Оценка моделей
    st.subheader("Оценка моделей")
    accuracy_log_reg, recall_log_reg, f1_log_reg = evaluate_model(sgd_model, X_test_selected, y_test, "Логистическая регрессия")
    accuracy_dt, recall_dt, f1_dt = evaluate_model(dt_model, X_test_selected, y_test, "Дерево решений")
    accuracy_rf, recall_rf, f1_rf = evaluate_model(rf_model, X_test_selected, y_test, "Случайный лес")
    
    # Создаем словарь для хранения метрик
    metrics_data = {
        "Модель": ["Логистическая регрессия", "Дерево решений", "Случайный лес"],
        "Accuracy": [accuracy_log_reg, accuracy_dt, accuracy_rf],
        "Recall": [recall_log_reg, recall_dt, recall_rf],
        "F1 Score": [f1_log_reg, f1_dt, f1_rf],
    }

    # Создаем DataFrame
    metrics_df = pd.DataFrame(metrics_data)

    # Выводим сводную таблицу
    st.subheader("Сводная таблица с метриками")
    st.dataframe(metrics_df)

    # Ввод текста для предсказания диагноза
    st.subheader("Введите симптомы для предсказания диагноза")
    input_data = st.text_area("Введите текст")

    if st.button("Предсказать диагноз"):
        if input_data:
            # Здесь предполагается, что у вас есть DataFrame mkb_df для кодов и названий диагнозов
            mkb_df = pd.read_csv('Datasets\МКБ_коды.csv')

            # Пример предсказания для каждой модели
            predicted_diagnosis_lg = predict_diagnosis(sgd_model, [input_data], tfidf_vectorizer, pca, selector, mkb_df)
            predicted_diagnosis_dt = predict_diagnosis(dt_model, [input_data], tfidf_vectorizer, pca, selector, mkb_df)
            predicted_diagnosis_rf = predict_diagnosis(rf_model, [input_data], tfidf_vectorizer, pca, selector, mkb_df)

            # Добавляем новые строки для фраз и предсказанных диагнозов
            metrics_data["Фраза"] = [input_data] * 3  # Используем одну и ту же фразу
            metrics_data["Предсказанный диагноз"] = [
                predicted_diagnosis_lg, predicted_diagnosis_dt, predicted_diagnosis_rf
            ]

            # Обновляем DataFrame
            updated_metrics_df = pd.DataFrame(metrics_data)

            # Выводим обновленную таблицу
            st.subheader("Таблица с результатами предсказаний")
            st.dataframe(updated_metrics_df)
        else:
            st.warning("Пожалуйста, введите текст для предсказания.")

# Функция для оценки моделей
def evaluate_model(model, X_test_selected, y_test, model_name):
    y_pred = model.predict(X_test_selected)
    
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    return accuracy, recall, f1