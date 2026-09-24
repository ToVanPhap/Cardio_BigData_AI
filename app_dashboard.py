import streamlit as st
import pandas as pd
import mysql.connector
import plotly.express as px
from config import DB_CONFIG

from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.feature import VectorAssembler

STREAMLIT_CONFIG = {
    'host': DB_CONFIG['host'],
    'user': DB_CONFIG['user'],
    'password': DB_CONFIG['password'],
    'database': DB_CONFIG['database']
}

st.set_page_config(page_title="Hệ Thống Chẩn Đoán Tim Mạch", page_icon="🫀", layout="wide")

st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined" rel="stylesheet" />
    <style>
        .stApp { background-color: #F8FBFA; }
        .medical-header { color: #0A4D68; font-weight: 600; display: flex; align-items: center; gap: 10px; padding-bottom: 15px; border-bottom: 2px solid #E0EBEB; margin-bottom: 25px; }
        .google-icon { font-size: 32px !important; color: #088395; }
        .risk-high { color: #D71313; font-weight: bold; font-size: 24px; }
        .risk-low { color: #088395; font-weight: bold; font-size: 24px; }
        .rec-box { background-color: white; padding: 15px; border-radius: 8px; border-left: 5px solid #088395; box-shadow: 0 2px 4px rgba(0,0,0,0.05); margin-bottom: 10px; }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_spark_and_model():
    spark = SparkSession.builder.appName("Medical_AI_Dashboard").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    model = RandomForestClassificationModel.load("hdfs://localhost:9000/random_forest_model")
    return spark, model

@st.cache_data
def load_mysql_data(query):
    conn = mysql.connector.connect(**STREAMLIT_CONFIG)
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# ================= HỆ CHUYÊN GIA TẠO KHUYẾN NGHỊ ĐỘNG =================
def generate_recommendations(prob, ap_hi, ap_lo, bmi, cholesterol, gluc, smoke, active):
    recs = []
    
    if ap_hi >= 140 or ap_lo >= 90:
        recs.append("🩺 **Huyết áp:** Đang ở mức nguy hiểm. Cần theo dõi huyết áp hàng ngày, tuyệt đối ăn nhạt (giảm muối) và cân nhắc dùng thuốc theo đơn của bác sĩ.")
    elif ap_hi >= 130 or ap_lo >= 85:
        recs.append("🩺 **Huyết áp:** Hơi cao. Hãy điều chỉnh chế độ ăn uống ngay để ngăn ngừa tiến triển thành bệnh tăng huyết áp.")
        
    if bmi >= 25:
        recs.append(f"⚖️ **Cân nặng:** Chỉ số BMI của bạn là {bmi:.1f} (Thừa cân/Béo phì). Giảm 5-10% trọng lượng cơ thể sẽ giúp giảm hàng tấn áp lực lên tim mạch.")
        
    if cholesterol > 1 or gluc > 1:
        recs.append("🩸 **Sinh hóa máu:** Mức Cholesterol hoặc Đường huyết đang cao. Cần hạn chế tối đa mỡ động vật, đồ ngọt, tăng cường chất xơ từ rau xanh.")
        
    if smoke:
        recs.append("🚭 **Thuốc lá:** Khói thuốc là nguyên nhân trực tiếp làm tắc nghẽn mạch máu. Cai thuốc lá là hành động cứu sống trái tim bạn lúc này.")
    if not active:
        recs.append("🏃 **Vận động:** Trái tim cần được rèn luyện. Hãy bắt đầu bằng những bài tập nhẹ nhàng như đi bộ 30 phút mỗi ngày.")
        
    if len(recs) == 0 and prob < 50:
        recs.append("✨ **Tình trạng lý tưởng:** Mọi chỉ số của bạn đều vô cùng tuyệt vời. Trái tim bạn đang rất khỏe mạnh, hãy tiếp tục duy trì lối sống này nhé!")
        
    return recs

st.markdown("<h1 class='medical-header'><span class='material-symbols-outlined google-icon' style='font-size: 42px !important;'>monitor_heart</span> TRUNG TÂM PHÂN TÍCH & CHẨN ĐOÁN TIM MẠCH</h1>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📊 Báo Cáo Phân Tích (MySQL)", "🤖 Bác Sĩ AI Chẩn Đoán (Spark MLlib)"])

# ================= TAB 1 ĐÃ ĐƯỢC CẬP NHẬT 4 BIỂU ĐỒ =================
with tab1:
    try:
        df_bp = load_mysql_data("SELECT * FROM risk_bp")
        df_age = load_mysql_data("SELECT * FROM risk_age")
        df_chol = load_mysql_data("SELECT * FROM risk_cholesterol")
        df_bmi = load_mysql_data("SELECT * FROM risk_bmi")

        # --- HÀNG 1: TOP 1 & TOP 2 (Huyết áp & Tuổi tác) ---
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<h3 class='medical-header' style='font-size: 20px;'><span class='material-symbols-outlined google-icon'>blood_pressure</span> 1. Cảnh báo rủi ro theo Huyết áp</h3>", unsafe_allow_html=True)
            fig_bp = px.bar(df_bp, x='bp_category', y='risk_percentage', text='risk_percentage', color='risk_percentage', color_continuous_scale=['#05BFDB', '#F8F988', '#FF6D60', '#D71313'])
            fig_bp.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
            fig_bp.update_layout(plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=10, l=10, r=10, b=10))
            st.plotly_chart(fig_bp, use_container_width=True)

        with col2:
            st.markdown("<h3 class='medical-header' style='font-size: 20px;'><span class='material-symbols-outlined google-icon'>vital_signs</span> 2. Diễn tiến rủi ro theo Độ tuổi</h3>", unsafe_allow_html=True)
            fig_age = px.line(df_age, x='age_group', y='risk_percentage', markers=True, text='risk_percentage')
            fig_age.update_traces(line=dict(color='#0A4D68', width=3), marker=dict(size=12, color='#FF6D60', line=dict(width=2, color='white')), texttemplate='%{text:.2f}%', textposition="top center")
            fig_age.update_layout(plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=10, l=10, r=10, b=10))
            st.plotly_chart(fig_age, use_container_width=True)
            
        st.markdown("<br>", unsafe_allow_html=True)

        # --- HÀNG 2: TOP 3 & TOP 4 (Cholesterol & BMI) ---
        col3, col4 = st.columns(2)
        with col3:
            st.markdown("<h3 class='medical-header' style='font-size: 20px;'><span class='material-symbols-outlined google-icon'>water_drop</span> 3. Cảnh báo rủi ro theo Cholesterol</h3>", unsafe_allow_html=True)
            fig_chol = px.bar(df_chol, x='category', y='risk_percentage', text='risk_percentage', color='risk_percentage', color_continuous_scale=['#05BFDB', '#F8F988', '#D71313'])
            fig_chol.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
            fig_chol.update_layout(
                plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=10, l=10, r=10, b=10),
                xaxis=dict(tickvals=[1,2,3], ticktext=['1. Bình thường', '2. Cao', '3. Rất cao'])
            )
            st.plotly_chart(fig_chol, use_container_width=True)

        with col4:
            st.markdown("<h3 class='medical-header' style='font-size: 20px;'><span class='material-symbols-outlined google-icon'>accessibility_new</span> 4. Phân bổ rủi ro theo Chỉ số BMI</h3>", unsafe_allow_html=True)
            fig_bmi = px.pie(df_bmi, names='bmi_category', values='sick_people', hole=0.5, color='bmi_category', color_discrete_map={'1. Thieu can': '#05BFDB', '2. Binh thuong': '#088395', '3. Thua can': '#F8F988', '4. Beo phi': '#FF6D60'})
            fig_bmi.update_traces(textposition='inside', textinfo='percent+label')
            fig_bmi.update_layout(margin=dict(t=10, l=10, r=10, b=10))
            st.plotly_chart(fig_bmi, use_container_width=True)

    except Exception as e:
        st.error(f"🔌 Lỗi kết nối CSDL: {e}")

# ================= TAB 2  =================
with tab2:
    st.markdown("<h3 class='medical-header' style='font-size: 20px;'><span class='material-symbols-outlined google-icon'>smart_toy</span> Nhập Chỉ Số Bệnh Nhân</h3>", unsafe_allow_html=True)
    
    with st.form("ai_predict_form"):
        c1, c2, c3, c4 = st.columns(4)
        age = c1.number_input("Tuổi", min_value=1, max_value=120, value=50)
        gender = c2.selectbox("Giới tính", options=[1, 2], format_func=lambda x: "Nữ" if x==1 else "Nam")
        height = c3.number_input("Chiều cao (cm)", min_value=50, max_value=250, value=165)
        weight = c4.number_input("Cân nặng (kg)", min_value=10, max_value=200, value=65)

        c5, c6, c7, c8 = st.columns(4)
        ap_hi = c5.number_input("Huyết áp Tâm thu (ap_hi)", min_value=50, max_value=250, value=120)
        ap_lo = c6.number_input("Huyết áp Tâm trương (ap_lo)", min_value=30, max_value=150, value=80)
        cholesterol = c7.selectbox("Cholesterol", options=[1, 2, 3], format_func=lambda x: "1: Bình thường" if x==1 else ("2: Cao" if x==2 else "3: Rất cao"))
        gluc = c8.selectbox("Đường huyết (Gluc)", options=[1, 2, 3], format_func=lambda x: "1: Bình thường" if x==1 else ("2: Cao" if x==2 else "3: Rất cao"))

        c9, c10, c11 = st.columns(3)
        smoke = c9.checkbox("Hút thuốc")
        alco = c10.checkbox("Uống rượu bia")
        active = c11.checkbox("Có tập thể dục", value=True)

        submitted = st.form_submit_button("🤖 Yêu cầu AI Chẩn đoán")

    if submitted:
        with st.spinner("Đang chuẩn đoán..."):
            try:
                spark, model = load_spark_and_model()
                bmi = weight / ((height/100)**2)

                # TỐI ƯU TỐC ĐỘ: Khai báo tĩnh tên cột thay vì load từ HDFS
                feature_columns = ['gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'age_years', 'bmi']
                input_data = [(int(gender), float(height), float(weight), float(ap_hi), float(ap_lo), 
                               int(cholesterol), int(gluc), int(smoke), int(alco), int(active), 
                               int(age), float(bmi))]
                
                patient_df = spark.createDataFrame(input_data, schema=feature_columns)
                assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
                patient_ml = assembler.transform(patient_df)

                prediction = model.transform(patient_ml)
                
                result = prediction.select("prediction", "probability").collect()[0]
                pred_label = result["prediction"]
                prob = result["probability"][1] * 100 

                st.markdown("---")
                if pred_label == 1:
                    st.markdown(f"<p class='risk-high'>⚠️ CẢNH BÁO: Phát hiện rủi ro Bệnh Tim Mạch!</p>", unsafe_allow_html=True)
                    st.error(f"Xác suất mắc bệnh do AI đánh giá: {prob:.2f}%")
                else:
                    st.markdown(f"<p class='risk-low'>✅ AN TOÀN: Trái tim đang trong vùng an toàn.</p>", unsafe_allow_html=True)
                    st.success(f"Xác suất mắc bệnh do AI đánh giá: {prob:.2f}%")
                
                # HIỂN THỊ KHUYẾN NGHỊ ĐA DẠNG
                st.markdown("<h4 style='color: #0A4D68; margin-top: 20px;'>📋 Phân tích & Lời khuyên chi tiết:</h4>", unsafe_allow_html=True)
                advices = generate_recommendations(prob, ap_hi, ap_lo, bmi, cholesterol, gluc, smoke, active)
                
                for advice in advices:
                    st.markdown(f"<div class='rec-box'>{advice}</div>", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Lỗi: {e}. Đảm bảo Hadoop HDFS đang bật (chạy lệnh jps).")