from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, Ridge
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

app = FastAPI()

# Đọc dữ liệu từ tệp CSV
df = pd.read_csv("Gia_Vang_2019_2022.csv")
df.columns = ["Date", "Price", "Open", "Vol"]

# 1. Biểu đồ phân bổ dữ liệu:
# Tạo dữ liệu mẫu để tạo bảng phân bổ dữ liệu
data = {
    "Date": df["Date"].values,
    "Price": df["Price"].values,
    "Open": df["Open"].values,
    "Vol": df["Vol"].values
}

# Biểu đồ Pair Plot cho tất cả các biến
sns.pairplot(df[["Price", "Open", "Vol"]])
plt.show()

# Chuyển đổi tất cả các cột thành mảng NumPy
x1 = np.array(df["Open"].values, dtype=float)
x2 = np.array(df["Vol"].values, dtype=float)
y = np.array(df["Price"].values, dtype=float)

# Số lượng quan sát
N = len(y)
alpha = 1.0  # Tham số điều chỉnh
X = np.column_stack((x1, x2))

# Hồi quy tuyến tính
y_pred1 = np.zeros_like(y)
m1 = (N * np.sum(x1 * y) - np.sum(x1) * np.sum(y)) / (N * np.sum(x1 ** 2) - (np.sum(x1) ** 2)) 
y_pred1 += m1 * x1
m2 = (N * np.sum(x2 * (y - y_pred1)) - np.sum(x2) * np.sum(y - y_pred1)) / (N * np.sum(x2 ** 2) - (np.sum(x2) ** 2))
b = np.mean(y) - (m1 * np.mean(x1) + m2 * np.mean(x2))

# Hàm dự đoán của hồi quy tuyến tính
def predict_gold_price(open_value: float, vol_value: float) -> float:
    return m1 * open_value + m2 * vol_value + b

# Các hàm tính MSE và R²
def calculate_mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def calculate_r2(y_true, y_pred):
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    return 1 - (ss_residual / ss_total)

# Dự đoán giá trị y (giá vàng) dựa vào x1 và x2
y_pred_linear = m1 * x1 + m2 * x2 + b
mse_linear = calculate_mse(y, y_pred_linear)
r2_linear = calculate_r2(y, y_pred_linear)

# Hồi quy Lasso
lasso_m1 = (N * np.sum(x1 * y) - np.sum(x1) * np.sum(y)) / (N * np.sum(x1 ** 2) + alpha - (np.sum(x1) ** 2))
lasso_y_pred1 = lasso_m1 * x1
lasso_m2 = (N * np.sum(x2 * (y - lasso_y_pred1)) - np.sum(x2) * np.sum(y - lasso_y_pred1)) / (N * np.sum(x2 ** 2) + alpha - (np.sum(x2) ** 2))
lasso_b = np.mean(y) - (lasso_m1 * np.mean(x1) + lasso_m2 * np.mean(x2))

# Hàm dự đoán cho hồi quy Lasso
def predict_gold_price_lasso(open_value: float, vol_value: float) -> float:
    return lasso_m1 * open_value + lasso_m2 * vol_value + lasso_b

# Dự đoán giá trị y cho hồi quy Lasso
lasso_y_pred = lasso_m1 * x1 + lasso_m2 * x2 + lasso_b
mse_lasso = calculate_mse(y, lasso_y_pred)
r2_lasso = calculate_r2(y, lasso_y_pred)

# Sử dụng hàm Lasso trong thư viện scikit-learn
lasso_model = Lasso(alpha=alpha, max_iter=1000)
lasso_model.fit(X, y)

# Hàm dự đoán cho Lasso
def predict_gold_price_lasso_sklearn(open_value: float, vol_value: float) -> float:
    return lasso_model.predict(np.array([[open_value, vol_value]]))[0]

# Neural Network Regression với ReLU
neural_model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, activation='relu', solver='adam', random_state=0)
neural_model.fit(X, y)

# Hàm dự đoán cho Neural Network (Đã sửa lỗi)
def predict_gold_price_neural(open_value: float, vol_value: float) -> float:
    return neural_model.predict(np.array([[open_value, vol_value]]))[0]  # Lấy giá trị đầu tiên từ mảng kết quả

# Dự đoán giá trị y cho Neural Network
neural_y_pred = neural_model.predict(X)
mse_neural = calculate_mse(y, neural_y_pred)
r2_neural = calculate_r2(y, neural_y_pred)

# 3: Phương pháp Bagging cho 3 mô hình hồi quy
# Tải dữ liệu mẫu
dulieu = load_iris()
X, y = dulieu.data, dulieu.target

# Thêm các biến x1, x2
x1 = X[:, 0]  # Giả sử x1 là cột đầu tiên của X
x2 = X[:, 1]  # Giả sử x2 là cột thứ hai của X

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Tạo mô hình Bagging với cây quyết định
bagging_model = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=50, random_state=42)

# Huấn luyện mô hình
bagging_model.fit(X_train, y_train)

# Dự đoán và đánh giá mô hình Bagging
y_pred = bagging_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Định nghĩa lớp dữ liệu đầu vào
class PredictionInput(BaseModel):
    open: float
    vol: float

# Endpoint giá vàng
@app.post("/predict")
async def predict(input_data: PredictionInput):
    open_value = input_data.open
    vol_value = input_data.vol

    try:
        # Dự đoán giá vàng
        predicted_linear = predict_gold_price(open_value, vol_value)
        predicted_lasso = predict_gold_price_lasso_sklearn(open_value, vol_value)
        predicted_neural = predict_gold_price_neural(open_value, vol_value)

        return {
            "predicted_linear": predicted_linear,
            "predicted_lasso": predicted_lasso,
            "predicted_neural": predicted_neural,
            "mse_linear": mse_linear,
            "r2_linear": r2_linear,
            "mse_lasso": mse_lasso,
            "r2_lasso": r2_lasso,
            "mse_neural": mse_neural,
            "r2_neural": r2_neural
        }
    except Exception as e:
        return {"error": str(e)}

# Trang chính với form nhập liệu
@app.get("/", response_class=HTMLResponse)
async def get_form():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Dự đoán giá vàng</title>
        <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
        <style>
            body {
                display: flex;
                justify-content: center;
                align-items: center;
                min-height: 100vh;
                grid-template-columns: 1fr 2fr;
                gap: 20px;
                background-color: #f8f8f8; /* Màu nền nhạt */
            }

            .container {
                display: grid;
                grid-template-columns: 1fr 2fr;
                gap: 20px;
            }

            .form-container {
                background-color: #f0f0f0;
                padding: 20px;
                border-radius: 5px;
                text-align: center; /* Căn giữa nội dung form */
            }

            .result-container {
                background-color: #fff;
                padding: 20px;
                border-radius: 5px;
            }

        </style>
    </head>
    <body>
        <div class="container">
            <div class="form-container">
                <h2 class="card-title text-center">Dự đoán giá vàng</h2>
                <form id="predictionForm">
                    <div class="form-group">
                        <label for="open">Giá mở cửa:</label>
                        <input type="number" id="open" name="open" class="form-control" step="any" required>
                    </div>
                    <div class="form-group">
                        <label for="vol">VoL (K):</label>
                        <input type="number" id="vol" name="vol" class="form-control" step="any" required>
                    </div>
                    <button type="button" class="btn btn-primary" onclick="predict()">Dự đoán</button>
                </form>
            </div>

            <div class="result-container">
                <div id="result"></div>
            </div>
        </div>

        <script>
        async function predict() {
            const open = document.getElementById('open').value;
            const vol = document.getElementById('vol').value;

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        open: parseFloat(open),
                        vol: parseFloat(vol)
                    }),
                });

                if (!response.ok) {
                    throw new Error('Mã phản hồi không hợp lệ');
                }

                const data = await response.json();
                document.getElementById('result').innerHTML = `
                    <h4>Kết quả dự đoán:</h4>
                    <p>Giá vàng dự đoán theo Hồi quy tuyến tính: ${data.predicted_linear.toFixed(2)} USD</p>
                    <p>Giá vàng dự đoán theo Hồi quy Lasso: ${data.predicted_lasso.toFixed(2)} USD</p>
                    <p>Giá vàng dự đoán theo Neural Network (ReLU): ${data.predicted_neural.toFixed(2)} USD</p>
                    <div>
                        <h5>MSE và R²:</h5>
                        <p>MSE Hồi quy tuyến tính: ${data.mse_linear.toFixed(2)}</p>
                        <p>R² Hồi quy tuyến tính: ${data.r2_linear.toFixed(2)}</p>
                        <p>MSE Hồi quy Lasso: ${data.mse_lasso.toFixed(2)}</p>
                        <p>R² Hồi quy Lasso: ${data.r2_lasso.toFixed(2)}</p>
                        <p>MSE Neural Network (ReLU): ${data.mse_neural.toFixed(2)}</p>
                        <p>R² Neural Network (ReLU): ${data.r2_neural.toFixed(2)}</p>
                    </div>
                `;
            } catch (error) {
                document.getElementById('result').innerHTML = `
                    <p class="text-danger">Đã xảy ra lỗi: ${error.message}</p>
                `;
            }
        }
        </script>
    </body>
    </html>
    """

# 4: Confusion matrix 
#Tạo 1 dataframe từ thư viện pandas với đối tượng data
df = pd.DataFrame(data)

#Tạo confusion matrix
cm = confusion_matrix(y_test, y_pred)

#Hiển thị confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dulieu.target_names)
disp.plot()
plt.show()
