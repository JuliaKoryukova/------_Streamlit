# streamlit_app.py
import streamlit as st
from pathlib import Path

# Импортирование страниц приложения
from hello_page import run
from analize_page import show_analize_page
from models_page import show_models_page  # Импортируем новую страницу

# Главная функция для запуска приложения
def main():
    # Заголовок приложения
    st.title("Медицинский помощник ИИ")
    
    # Добавление меню навигации в боковую панель
    st.sidebar.title("Навигация")
    
    # Переключатель для навигации между страницами
    page = st.sidebar.selectbox("Выберите страницу", ("Приветствие", "Анализ данных", "Модели"))
    
    # Логика для отображения выбранной страницы
    if page == "Приветствие":
        run()  # Открывает страницу приветствия
    elif page == "Анализ данных":
        show_analize_page()  # Открывает страницу анализа данных
    elif page == "Модели":
        show_models_page()  # Открывает страницу с моделями

# Запуск основного приложения
if __name__ == "__main__":
    main()