import numpy as np
import matplotlib.pyplot as plt       
from sklearn.linear_model import LinearRegression
#lostfunction
def caculate_mse(y, y_hat):
    e = y_hat-y
    return np.sum(e*e)/len(y)
#build update
def gradient_descent (x, y, lr, epochs):
    n = len(y)
    w = 0.0
    b = 0.0 
    loss_history = []
    for i in range (epochs):
        y_hat = w*x + b
        loss = caculate_mse(y,y_hat)
        dw = 2 * np.mean(x*(y_hat - y))
        db = 2 * np.mean(y_hat-y)
        w = w -lr*dw
        b = b -lr*db
        loss_history.append(loss)
    return w, b, loss_history
if __name__ == "__main__":
    np.random.seed(42) 
    x = 2 * np.random.rand(100)
    y = 2 * x + 3 + np.random.randn(100)
    
    w_tim_duoc, b_tim_duoc, losses = gradient_descent(x, y, lr=0.1, epochs=500)

    model = LinearRegression()
    model.fit(x.reshape(-1, 1), y)
    w_sklearn = model.coef_[0]
    b_sklearn = model.intercept_

    print(f"Của bạn: w = {w_tim_duoc:.4f}, b = {b_tim_duoc:.4f}")
    print(f"Sklearn: w = {w_sklearn:.4f}, b = {b_sklearn:.4f}")

    #vẽ đồ thị
    plt.figure(figsize=(8, 5))

    # Vẽ các điểm dữ liệu mẫu
    plt.scatter(x, y, color='blue', alpha=0.5, label='Dữ liệu thực tế')

    # Vẽ đường của bạn (màu đỏ)
    plt.plot(x, w_tim_duoc * x + b_tim_duoc, color='red', linewidth=2, label='Của tôi (GD)')

    # Vẽ đường của Sklearn (màu xanh lá, nét đứt)
    plt.plot(x, w_sklearn * x + b_sklearn, color='green', linestyle='--', linewidth=2, label='Sklearn')

    # Trang trí đồ thị
    plt.title("So sánh Linear Regression: Tự code vs Thư viện")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()  # Hiện bảng chú thích các đường
    plt.grid(True)
    # Thêm dòng này để lưu ảnh:
    plt.savefig("so_sanh.png")
    plt.show()
    # Vẽ biểu đồ Loss
    plt.figure(figsize=(8, 4))
    plt.plot(losses, color='red', linewidth=2)
    plt.title("Quá trình Loss (MSE) giảm dần theo từng Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig("loss_evolution.png")