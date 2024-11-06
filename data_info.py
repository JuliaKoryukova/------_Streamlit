#data_info.py
import streamlit as st
from models_page import get_dataframe

import seaborn as sns
from matplotlib import pyplot as plt
# importing libraries
import altair as alt

from urllib.error import URLError

df = get_dataframe()
st.title('Предварительная обработка данных')

st.table(df.head(10))

