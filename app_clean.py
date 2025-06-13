import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 加载模型与标准化器
gbdt_model = joblib.load('gbdt_model.joblib')
scaler = joblib.load('scaler.joblib')

# 页面设置
st.set_page_config(page_title="AIP-MDA5-ILD预测系统", page_icon="🫁", layout="centered")
st.title("AIP-MDA5-ILD分级预测")
st.markdown("""
本工具用于预测 MDA5阳性皮肌炎患者是否存在重度间质性肺疾病（ILD）。
请根据化验结果输入以下生物指标，系统将自动给出预测结果。
""")

with st.container():
    st.header("🔬 输入检测数据")
    fibrinogen = st.number_input("纤维蛋白原（g/L）", min_value=0.0)
    hb_albumin_ratio = st.number_input("血红蛋白 ÷ 白蛋白", min_value=0.0)
    triglyceride = st.number_input("甘油三酯（mmol/L）", min_value=0.0)
    anti_ro52 = st.number_input("抗RO52滴度", min_value=0.0)
    ldh = st.number_input("LDH（U/L）", min_value=0.0)
    antibody = st.selectbox("抗合成酶抗体阳性", [0, 1])
    wbc = st.number_input("白细胞计数（×10^9/L）", min_value=0.0)

input_data = {
    '纤维蛋白原': fibrinogen,
    '血红蛋白_÷_白蛋白': hb_albumin_ratio,
    '甘油三酯': triglyceride,
    '抗RO52滴度': anti_ro52,
    'LDH': ldh,
    '抗合成酶抗体阳性': antibody,
    '白细胞计数': wbc
}

X = pd.DataFrame([input_data])
X_scaled = scaler.transform(X)
prob = gbdt_model.predict_proba(X_scaled)[0][1]
prediction = "ILD分级为 1 级（重度）" if prob >= 0.5 else "ILD分级为 0 级（非重度）"

st.markdown("## 🧠 预测结果")
st.success(f"{prediction}")
st.info(f"预测概率：**{prob:.3f}**")
