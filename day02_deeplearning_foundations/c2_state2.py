import numpy as np
def parameters(n_x, n_h, n_y):
    np.random.seed(42)

    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))

    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))
    parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }
    return parameters

def sigmoid(z):
    z_clip = np.clip(z, -500, 500)
    s = 1.0/(1.0+np.exp(-z_clip))
    return s

def forward_propagation(X, parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)

    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    cache = {
        "Z1": Z1,
        "A1": A1,
        "Z2": Z2,
        "A2": A2
    }
    return A2, cache

def compute_cost(A2, Y):
    m = Y.shape[1]
    eps = 1e-15 # Tránh lỗi log(0)
    cost = - (1.0 / m) * np.sum(Y * np.log(A2 + eps) + (1.0 - Y) * np.log(1.0 - A2 + eps))
    return float(np.squeeze(cost))

def load_planar_dataset(seed=42):
    np.random.seed(seed)
    m = 400            # 400 mẫu dữ liệu
    N = int(m / 2)     # 200 điểm nhãn 0, 200 điểm nhãn 1
    D = 2              # 2 đặc trưng tọa độ (x1, x2)
    X = np.zeros((m, D))
    Y = np.zeros((m, 1), dtype='uint8')
    a = 4              # Độ dài tối đa của cánh hoa

    for j in range(2):
        ix = range(N * j, N * (j + 1))
        # t là góc xoay, r là bán kính uốn lượn theo hàm sin tạo thành 4 cánh hoa
        t = np.linspace(j * 3.12, (j + 1) * 3.12, N) + np.random.randn(N) * 0.2
        r = a * np.sin(4 * t) + np.random.randn(N) * 0.2
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        Y[ix] = j

    # Chuyển vị .T để đưa về Shape chuẩn của Andrew Ng:
    # X: (số đặc trưng, số mẫu) = (2, 400)
    # Y: (1, số mẫu)            = (1, 400)
    return X.T, Y.T

def backward_propagation(parameters, cache, X, Y):
    m = X.shape[1]

    W1 = parameters["W1"]
    W2 = parameters["W2"]
    A1 = cache["A1"]
    A2 = cache["A2"]
    dZ2 = A2 - Y
    dW2 = (1.0/m) * np.dot(dZ2, A1.T)
    db2 = (1.0/m) * np.sum(dZ2, axis=1, keepdims=True)

    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
    dW1 = (1.0/m) * np.dot(dZ1, X.T)
    db1 = (1.0/m) * np.sum(dZ1, axis=1, keepdims=True)

    assert dW2.shape == W2.shape, f"Shape dW2 {dW2.shape} khác W2 {W2.shape}"
    assert db2.shape == b2.shape, f"Shape db2 {db2.shape} khác b2 {b2.shape}"
    assert dW1.shape == W1.shape, f"Shape dW1 {dW1.shape} khác W1 {W1.shape}"
    assert db1.shape == b1.shape, f"Shape db1 {db1.shape} khác b1 {b1.shape}"

    grads = {
        "dW1": dW1,
        "db1": db1,
        "dW2": dW2,
        "db2": db2
    }
    return grads

def update_parameters(parameters, grads, learning_rate=1.2):
    parameters["W1"] = parameters["W1"] - learning_rate * grads["dW1"]
    parameters["b1"] = parameters["b1"] - learning_rate * grads["db1"]
    parameters["W2"] = parameters["W2"] - learning_rate * grads["dW2"]
    parameters["b2"] = parameters["b2"] - learning_rate * grads["db2"]
    return parameters

def nn_model(X, Y, n_h=4, num_iterations=10000, learning_rate=1.2, print_cost=True):
    np.random.seed(3)
    n_x = X.shape[0] # Số đặc trưng = 2
    n_y = Y.shape[0] # Số đầu ra = 1

    # 1. Khởi tạo trọng số phá vỡ tính đối xứng
    params = parameters(n_x, n_h, n_y)
    costs = []

    # 2. Vòng lặp huấn luyện Gradient Descent
    for i in range(num_iterations):
        # Bước a: Lan truyền xuôi (Forward)
        A2, cache = forward_propagation(X, params)

        # Bước b: Tính chi phí sai lệch (Cost)
        cost = compute_cost(A2, Y)

        # Bước c: Lan truyền ngược tính gradient (Backward)
        grads = backward_propagation(params, cache, X, Y)

        # Bước d: Cập nhật trọng số mới (Update)
        params = update_parameters(params, grads, learning_rate=learning_rate)

        # In kết quả theo dõi mỗi 1000 vòng lặp
        if i % 1000 == 0:
            costs.append(cost)
            if print_cost:
                print(f"Epoch {i:5d} | Cost: {cost:.5f}")

    return params, costs

def predict(parameters, X):
    A2, _ = forward_propagation(X, parameters)
    predictions = (A2 > 0.5).astype(int)
    return predictions

if __name__ == "__main__":
    # 1. Nạp dữ liệu cánh hoa
    X, Y = load_planar_dataset()

    # 2. Huấn luyện mô hình với 4 nơ-ron ở tầng ẩn
    params, costs = nn_model(X, Y, n_h=4, num_iterations=10000, learning_rate=1.2)

    # 3. Đánh giá độ chính xác
    predictions = predict(params, X)
    accuracy = float(np.mean(predictions == Y) * 100)
    print(f"\n[+] Độ chính xác trên tập Planar: {accuracy:.2f}%")