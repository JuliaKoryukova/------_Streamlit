# Analize_page.py
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from wordcloud import WordCloud

# Функции для загрузки данных
@st.cache_data
def load_data():
    # Загружаем основной файл с данными
    med_df = pd.read_csv('Datasets\data_cleaned.csv')  # Путь к файлу данных
    return med_df

@st.cache_data
def load_mkb_codes():
    # Загружаем таблицу с кодами МКБ
    df_mkd = pd.read_csv('Datasets\МКБ_коды.csv')  # Путь к файлу с МКБ кодами
    return df_mkd

# Функция для построения облака слов
def plot_wordcloud(text_data, title):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(text_data))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.tight_layout()
    st.pyplot(plt)

# Основная функция для отображения страницы анализа
def show_analize_page():
    st.title("Анализ медицинских данных")

    # Загружаем данные
    med_df = load_data()
    df_mkd = load_mkb_codes()

    if med_df.empty or df_mkd.empty:
        st.write("Данные не загружены или файл пуст.")
        return

    st.write("## Обзор данных")
    st.write(med_df.head())

    # Преобразование столбца new_event_time в тип datetime
    med_df['new_event_time'] = pd.to_datetime(med_df['new_event_time'], errors='coerce')

    # Создание нового столбца year_month
    med_df['year_month'] = med_df['new_event_time'].dt.to_period('M')

    # Раздел 1: Анализ симптомов
    if 'symptoms' in med_df.columns:
        st.write("### Анализ симптомов")

        # Облако слов для симптомов
        st.write("#### Облако слов для симптомов")
        symptoms_text = med_df.explode('symptoms')['symptoms'].dropna().tolist()
        plot_wordcloud(symptoms_text, "Облако слов для симптомов")

        # Топ-10 симптомов
        st.write("#### Топ-10 симптомов")
        symptom_frequency = med_df.explode('symptoms')['symptoms'].value_counts().reset_index()
        symptom_frequency.columns = ['symptoms', 'count']
        st.write(symptom_frequency.head(10))

        # Количество симптомов по месяцам
        st.write("#### Количество симптомов по месяцам")
        time_symptom = med_df.explode('symptoms').groupby(['year_month', 'symptoms']).size().reset_index(name='count')
        total_symptoms = time_symptom.groupby('year_month')['count'].sum().reset_index()
        total_symptoms['year_month'] = total_symptoms['year_month'].dt.to_timestamp()

        plt.figure(figsize=(12, 6))
        plt.plot(total_symptoms['year_month'], total_symptoms['count'], marker='o', linestyle='-', color='b')
        plt.title('Общее количество симптомов по месяцам')
        plt.xlabel('Дата (Год-Месяц)')
        plt.ylabel('Количество симптомов')
        plt.xticks(rotation=45)
        plt.grid()
        plt.tight_layout()
        st.pyplot(plt)

    else:
        st.write("Столбец 'symptoms' (симптомы) отсутствует в данных.")

    # Раздел 2: Анализ анамнеза
    if 'anamnesis' in med_df.columns:
        st.write("### Анализ анамнеза")

        # Облако слов для анамнеза
        st.write("#### Облако слов для анамнеза")
        anamnesis_text = med_df.explode('anamnesis')['anamnesis'].dropna().tolist()
        plot_wordcloud(anamnesis_text, "Облако слов для анамнеза")

        # Топ-10 анамнезов
        st.write("#### Топ-10 анамнезов")
        anamnesis_frequency = med_df.explode('anamnesis')['anamnesis'].value_counts().reset_index()
        anamnesis_frequency.columns = ['anamnesis', 'count']
        st.write(anamnesis_frequency.head(10))

        # Количество анамнезов по месяцам
        st.write("#### Количество анамнезов по месяцам")
        time_anamnesis = med_df.explode('anamnesis').groupby(['year_month', 'anamnesis']).size().reset_index(name='count')
        total_anamnesis = time_anamnesis.groupby('year_month')['count'].sum().reset_index()
        total_anamnesis['year_month'] = total_anamnesis['year_month'].dt.to_timestamp()

        plt.figure(figsize=(12, 6))
        plt.plot(total_anamnesis['year_month'], total_anamnesis['count'], marker='o', linestyle='-', color='g')
        plt.title('Общее количество анамнезов по месяцам')
        plt.xlabel('Дата (Год-Месяц)')
        plt.ylabel('Количество анамнезов')
        plt.xticks(rotation=45)
        plt.grid()
        plt.tight_layout()
        st.pyplot(plt)

    else:
        st.write("Столбец 'anamnesis' (анамнезы) отсутствует в данных.")

    # Раздел 3: Анализ диагнозов
    if 'icd10' in med_df.columns:
        st.write("### Анализ диагнозов")

        # Облако слов для диагнозов
        st.write("#### Облако слов для диагнозов")
        med_df = med_df.merge(df_mkd, how='left', left_on='icd10', right_on='Код')
        diagnosis_text = med_df['Название'].dropna().tolist()
        plot_wordcloud(diagnosis_text, "Облако слов для диагнозов")

        # Топ-10 диагнозов
        st.write("#### Топ-10 диагнозов")
        diagnosis_frequency = med_df.groupby('Название').size().reset_index(name='count')
        diagnosis_frequency = diagnosis_frequency.sort_values(by='count', ascending=False).head(10)
        st.write(diagnosis_frequency)

        # Количество диагнозов по месяцам
        st.write("#### Количество диагнозов по месяцам")
        time_diagnosis = med_df.groupby(['year_month', 'icd10', 'Название']).size().reset_index(name='count')
        total_diagnosis = time_diagnosis.groupby('year_month')['count'].sum().reset_index()
        total_diagnosis['year_month'] = total_diagnosis['year_month'].dt.to_timestamp()

        plt.figure(figsize=(12, 6))
        plt.plot(total_diagnosis['year_month'], total_diagnosis['count'], marker='o', linestyle='-', color='r')
        plt.title('Общее количество диагнозов по месяцам')
        plt.xlabel('Дата (Год-Месяц)')
        plt.ylabel('Количество диагнозов')
        plt.xticks(rotation=45)
        plt.grid()
        plt.tight_layout()
        st.pyplot(plt)

    else:
        st.write("Столбец 'icd10' (диагнозы) отсутствует в данных.")