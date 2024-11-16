
import streamlit as st
import numpy as np
import joblib

# 사전 학습된 모델 불러오기
model_path = "random_forest_workload_model.pkl"
loaded_model = joblib.load(model_path)

# Streamlit App Title
st.title("📦 집배업무 강도 예측 시스템")
st.markdown("""
    **💡 간단한 데이터를 입력하여 집배 업무 강도를 예측하세요.**  
    아래 입력 필드에 데이터를 입력한 후, **[업무 강도 예측]** 버튼을 클릭하세요.
    """)

# 구분선
st.divider()

# 레이아웃 설정: 입력 필드 및 버튼
col1, col2 = st.columns(2)

# 왼쪽 열: 입력 필드
with col1:
    st.subheader("📋 데이터 입력")
    general_volume = st.number_input('📄 일반통상 물량', min_value=0, value=0)
    registered_volume = st.number_input('📑 등기통상 물량', min_value=0, value=0)
    parcel_volume = st.number_input('📦 등기소포 물량', min_value=0, value=0)
    EMS_volume = st.number_input('✈️ EMS 물량', min_value=0, value=0)
    pickup_volume = st.number_input('🚚 픽업 물량', min_value=0, value=0)
    driving_distance = st.number_input('🚗 운행거리 (km)', min_value=0, value=0)

# 오른쪽 열: 예측 결과
with col2:
    st.subheader("💼 예측 결과")
    if st.button("✨ 업무 강도 예측"):
        # 입력 데이터를 배열 형태로 준비
        input_data = np.array([[general_volume, registered_volume, parcel_volume, EMS_volume, pickup_volume, driving_distance]])
        # 예측 수행
        prediction = loaded_model.predict(input_data)

        # 예측 결과 표시
        st.success("✅ 예측 완료!")
        st.metric(label="📊 예측된 업무 강도", value=f"{prediction[0]:.3f}")

# 하단 구분선
st.divider()

# 하단 설명 또는 추가 정보
st.markdown("""
    🔍 **사용자 안내:**  
    - 예측 모델은 랜덤포레스트 모델로 평균 제곱근 오차(RMSE)는 0.13입니다.
    - RMSE가 0.13이라는 것은, 모델이 예측한 값이 평균적으로 실제 값에서 0.13 단위 정도 벗어난다는 것을 의미합니다.
    - 약 30만개의 데이터를 학습하였습니다.
    """)
