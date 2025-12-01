import streamlit as st
import pandas as pd

from utils.data_loader import read_profile_excel
from utils.model_loader import load_models
from utils.predictors import predict_all_toxicities
from utils.scoring import proba_to_score
from utils.feature_list import load_feature_sets   # <-- важно

st.set_page_config(page_title="Моделирование", layout="wide")

# Загружаем фичи трёх моделей
feature_sets = load_feature_sets()
cardio_features = feature_sets["cardio"]
neuro_features  = feature_sets["neuro"]
hepato_features = feature_sets["hepato"]

st.title("🧪 Предсказание токсичности ЛП на модели Зебрафиш")

############################################################
with st.sidebar:
    st.header("📌 Просмотр признаков моделей")

    tabs = st.tabs(["Кардио", "Нейро", "Гепато"])

    with tabs[0]:
        st.subheader("Кардиотоксичность")
        st.markdown(f"Всего признаков: **{len(cardio_features)}**")
        selected_cardio = st.selectbox(
            "Выберите признак",
            options=cardio_features,
            key="cardio_feature_select"
        )
        st.write(f"Выбранный признак: **{selected_cardio}**")

    with tabs[1]:
        st.subheader("Нейротоксичность")
        st.markdown(f"Всего признаков: **{len(neuro_features)}**")
        selected_neuro = st.selectbox(
            "Выберите признак",
            options=neuro_features,
            key="neuro_feature_select"
        )
        st.write(f"Выбранный признак: **{selected_neuro}**")

    with tabs[2]:
        st.subheader("Гепатотоксичность")
        st.markdown(f"Всего признаков: **{len(hepato_features)}**")
        selected_hepato = st.selectbox(
            "Выберите признак",
            options=hepato_features,
            key="hepato_feature_select"
        )
        st.write(f"Выбранный признак: **{selected_hepato}**")

#####################################################################

uploaded = st.file_uploader("Загрузите Excel-файл", type=["xlsx"])

if uploaded:
    df = read_profile_excel(uploaded)
    models = load_models()
    st.success("Файл загружен.")

    # === список для накопления результатов ===
    results = []

    for _, row in df.iterrows():
        drug = row["Drug"]
        profile_dict = row.drop("Drug").to_dict()

        try:
            cardio_x = [float(profile_dict.get(f, 0.0)) for f in cardio_features]
            neuro_x  = [float(profile_dict.get(f, 0.0)) for f in neuro_features]
            hepato_x = [float(profile_dict.get(f, 0.0)) for f in hepato_features]
        except Exception as e:
            st.error(f"Ошибка при формировании признаков для препарата {drug}: {e}")
            continue

        cardio_p, neuro_p, hepato_p = predict_all_toxicities(
            models,
            cardio_x,
            neuro_x,
            hepato_x
        )

        cardio_score = proba_to_score(cardio_p)
        neuro_score  = proba_to_score(neuro_p)
        hepato_score = proba_to_score(hepato_p)
        total = cardio_score + neuro_score + hepato_score

        # === добавляем в таблицу ===
        results.append({
            "Препарат": drug,
            "Кардио": cardio_score,
            "Нейро": neuro_score,
            "Гепато": hepato_score,
            "Общая": total
        })

    # === выводим Heatmap-таблицу ===
    # === продвинутая токсикологическая таблица ===
    st.markdown("## 📊 Итоговая таблица оценки органотоксичности препаратов")

    df_res = pd.DataFrame(results)

    # ===== Кастомная окраска =====
    def color_toxic(val):
        """Окраска для отдельных токсичностей 0–10."""
        if val < 6:
            return "background-color: #c9f7c9;"   # зелёный
        elif val < 8:
            return "background-color: #fff6a5;"   # жёлтый
        else:
            return "background-color: #ffb3b3;"   # красный

    def color_total(val):
        """Окраска для общей токсичности 0–30."""
        if val < 10:
            return "background-color: #c9f7c9;"   # зелёный
        elif val < 18:
            return "background-color: #fff6a5;"   # жёлтый
        else:
            return "background-color: #ffb3b3;"   # красный

    styled = df_res.style.applymap(color_toxic, subset=["Кардио", "Нейро", "Гепато"]) \
                        .applymap(color_total, subset=["Общая"])

    col_table, col_legend = st.columns([4, 1])

    with col_table:
        st.dataframe(styled)

    with col_legend:
        st.markdown("""
        <div style="margin-left:10px; padding:10px; border:1px solid #ccc; border-radius:8px; width:180px;">
        <b>Шкала токсичности</b><br><br>

        <b>Частные токсичности (0–10)</b><br>
        <div style="height:12px; background:linear-gradient(to right, #c9f7c9, #fff6a5, #ffb3b3);"></div>
        <div style="display:flex; justify-content:space-between; font-size:12px; margin-top:2px;">
        <span>0</span><span>6</span><span>8</span><span>10</span>
        </div>
        <div style="font-size:11px; color:#555; line-height:1.2; margin-top:2px;">
            <div style="white-space:nowrap;">0 — нетоксичен</div>
            <div style="white-space:nowrap;">10 — наиболее токсичен</div>
        </div>
        <br><br>

        <b>Общая токсичность (0–30)</b><br>
        <div style="height:12px; background:linear-gradient(to right, #c9f7c9, #fff6a5, #ffb3b3);"></div>
        <div style="display:flex; justify-content:space-between; font-size:12px; margin-top:2px;">
        <span>0</span><span>10</span><span>18</span><span>30</span>
        </div>
        <div style="font-size:11px; color:#555; line-height:1.2; margin-top:2px;">
            <div style="white-space:nowrap;">0 — нетоксичен</div>
            <div style="white-space:nowrap;">30 — наиболее токсичен</div>
        </div>
        </div>

        """, unsafe_allow_html=True)


    # === КНОПКА СКАЧИВАНИЯ ===
    import io

    def to_excel(df):
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="toxicity")
        return output.getvalue()

    excel_data = to_excel(df_res)

    st.download_button(
        label="📥 Скачать таблицу (Excel)",
        data=excel_data,
        file_name="toxicity_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
