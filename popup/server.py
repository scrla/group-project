from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import datetime

app = Flask(__name__)
CORS(app)

# preprocessing.ipynb 에서 저장된 모델 불러오기
model = joblib.load('best_lgbm_pipeline.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    bidAmount = data.get('bidAmount')
    minBidPrice = data.get('minBidPrice')

    if bidAmount is None or minBidPrice is None:
        return jsonify({'error': '입찰가 또는 최저입찰가가 누락됨'}), 400

    # 입력값 구성: preprocessing.ipynb와 동일한 feature 구조로 입력
    input_df = pd.DataFrame([{
        '개찰일시': pd.Timestamp.now(),  # 모델이 이 값을 요구한다면 현재 시간 사용
        '대분류': '공사',                # 프론트에서 전송하거나 임시 하드코딩
        '중분류': '토목',                # 필요에 따라 변경
        '최저입찰가': minBidPrice
    }])

    # 예측 실행
    predicted_rate = model.predict(input_df)[0]

    return jsonify({'predicted_rate': round(predicted_rate, 2)})

if __name__ == '__main__':
    app.run(port=5001, debug=True)
