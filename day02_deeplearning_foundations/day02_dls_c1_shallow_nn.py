import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Dam bao in tieng Viet tren terminal Windows khong bi loi font
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# ==========================================
# 1. CAC HAM COT LOI BAN VUA VIET
# ==========================================

def parameters(n_x, n_h, n_y):
    np.random.seed(42)
    # Nhan 0.01 de tranh bao hoa ham tanh va triet tieu gradient
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))
    
    params = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }
    return params

def sigmoid(z):
    z_clip = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z_clip))

def forward_propagation(X, params):
    W1 = params["W1"]
    b1 = params["b1"]
    W2 = params["W2"]
    b2 = params["b2"]

    # Tang an: kich hoat tanh
    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)

    # Tang dau ra: kich hoat sigmoid
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
    eps = 1e-15
    cost = - (1.0 / m) * np.sum(Y * np.log(A2 + eps) + (1.0 - Y) * np.log(1.0 - A2 + eps))
    return float(np.squeeze(cost))

def backward_propagation(params, cache, X, Y):
    m = X.shape[1]

    W1 = params["W1"]
    b1 = params["b1"]
    W2 = params["W2"]
    b2 = params["b2"]
    A1 = cache["A1"]
    A2 = cache["A2"]
    
    # 1. Gradient tang ra
    dZ2 = A2 - Y
    dW2 = (1.0 / m) * np.dot(dZ2, A1.T)
    db2 = (1.0 / m) * np.sum(dZ2, axis=1, keepdims=True)
    
    # 2. Gradient tang an (dao ham tanh = 1 - A1^2)
    dZ1 = np.dot(W2.T, dZ2) * (1.0 - np.power(A1, 2))
    dW1 = (1.0 / m) * np.dot(dZ1, X.T)
    db1 = (1.0 / m) * np.sum(dZ1, axis=1, keepdims=True)

    # Kiem tra shape bang assert
    assert dW2.shape == W2.shape, f"Shape dW2 {dW2.shape} khac W2 {W2.shape}"
    assert db2.shape == b2.shape, f"Shape db2 {db2.shape} khac b2 {b2.shape}"
    assert dW1.shape == W1.shape, f"Shape dW1 {dW1.shape} khac W1 {W1.shape}"
    assert db1.shape == b1.shape, f"Shape db1 {db1.shape} khac b1 {b1.shape}"

    grads = {
        "dW1": dW1,
        "db1": db1,
        "dW2": dW2,
        "db2": db2
    }
    return grads

def update_parameters(params, grads, learning_rate=1.2):
    params["W1"] = params["W1"] - learning_rate * grads["dW1"]
    params["b1"] = params["b1"] - learning_rate * grads["db1"]
    params["W2"] = params["W2"] - learning_rate * grads["dW2"]
    params["b2"] = params["b2"] - learning_rate * grads["db2"]
    return params

def predict(params, X):
    A2, _ = forward_propagation(X, params)
    predictions = (A2 > 0.5).astype(int)
    return predictions, A2

# ==========================================
# 2. KHO DULIEU PHI TUYEN (DATASET REPOSITORY)
# ==========================================

def load_planar_flower(seed=42):
    """Dataset 1: Canh hoa 4 canh (Planar Flower)"""
    np.random.seed(seed)
    m = 400
    N = int(m / 2)
    D = 2
    X = np.zeros((m, D))
    Y = np.zeros((m, 1), dtype='uint8')
    a = 4
    for j in range(2):
        ix = range(N * j, N * (j + 1))
        t = np.linspace(j * 3.12, (j + 1) * 3.12, N) + np.random.randn(N) * 0.2
        r = a * np.sin(4 * t) + np.random.randn(N) * 0.2
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        Y[ix] = j
    return X.T, Y.T

def load_two_moons(seed=42):
    """Dataset 2: Hai nua vang trang khuyet long nhau (Two Moons)"""
    from sklearn.datasets import make_moons
    X, Y = make_moons(n_samples=400, noise=0.2, random_state=seed)
    return X.T, Y.reshape(1, -1)

def load_concentric_circles(seed=42):
    """Dataset 3: Hai vong tron dong tam (Concentric Circles)"""
    from sklearn.datasets import make_circles
    X, Y = make_circles(n_samples=400, noise=0.08, factor=0.5, random_state=seed)
    return X.T, Y.reshape(1, -1)

def load_two_spirals(seed=42):
    """Dataset 4: Xoan oc kep Archimedes (Two Spirals)"""
    np.random.seed(seed)
    n = 200
    theta = np.sqrt(np.random.rand(n)) * 2 * np.pi
    
    r_a = 2 * theta + np.pi
    data_a = np.array([np.cos(theta) * r_a + np.random.randn(n) * 0.2,
                       np.sin(theta) * r_a + np.random.randn(n) * 0.2])
    
    r_b = -2 * theta - np.pi
    data_b = np.array([np.cos(theta) * r_b + np.random.randn(n) * 0.2,
                       np.sin(theta) * r_b + np.random.randn(n) * 0.2])
    
    X = np.hstack((data_a, data_b))
    Y = np.hstack((np.zeros((1, n)), np.ones((1, n))))
    return X, Y

DATASET_DICT = {
    "flower": ("Canh hoa Planar Flower", load_planar_flower, 4, 1.2),
    "moons": ("Hai vang trang Two Moons", load_two_moons, 4, 1.2),
    "circles": ("Vong tron dong tam Circles", load_concentric_circles, 4, 1.2),
    "spiral": ("Xoan oc kep Twin Spirals", load_two_spirals, 10, 1.0)
}

# ==========================================
# 3. TRAINING PIPELINE & VISUALIZATION
# ==========================================

def nn_model(X, Y, n_h=4, num_iterations=10000, learning_rate=1.2, print_cost=True):
    np.random.seed(3)
    n_x = X.shape[0]
    n_y = Y.shape[0]

    params = parameters(n_x, n_h, n_y)
    costs = []

    for i in range(num_iterations):
        A2, cache = forward_propagation(X, params)
        cost = compute_cost(A2, Y)
        grads = backward_propagation(params, cache, X, Y)
        params = update_parameters(params, grads, learning_rate=learning_rate)

        if i % 1000 == 0:
            costs.append(cost)
            if print_cost:
                print(f"Epoch {i:5d} | Cost: {cost:.5f}")

    return params, costs

if __name__ == "__main__":
    # Cu phap: python day02_dls_c1_shallow_nn.py [flower|moons|circles|spiral] [so_noron_an]
    dataset_name = "circles" # Mac dinh chay circles
    if len(sys.argv) > 1 and sys.argv[1].lower() in DATASET_DICT:
        dataset_name = sys.argv[1].lower()
    
    label_vn, loader_func, default_nh, default_lr = DATASET_DICT[dataset_name]
    
    n_h = default_nh
    if len(sys.argv) > 2:
        try:
            n_h = int(sys.argv[2])
        except ValueError:
            pass

    print(f"[*] Chay Shallow Neural Network tren tap: [{label_vn.upper()}]")
    print(f"[i] Cau hinh: n_h = {n_h} noron an | learning_rate = {default_lr}")
    
    X, Y = loader_func()
    print(f"[i] Dataset shape: X = {X.shape}, Y = {Y.shape}")

    # Huan luyen
    num_iterations = 10000
    params, costs = nn_model(X, Y, n_h=n_h, num_iterations=num_iterations, learning_rate=default_lr)

    # Danh gia
    predictions, _ = predict(params, X)
    accuracy = float(np.mean(predictions == Y) * 100)
    print(f"\n[+] Huan luyen hoan tat! Do chinh xac (Accuracy): {accuracy:.2f}%\n")

    # ==========================================
    # VE BIEU DO
    # ==========================================
    plt.figure(figsize=(14, 6))

    # 1. Learning curve
    plt.subplot(1, 2, 1)
    epochs_axis = np.arange(len(costs)) * 1000
    plt.plot(epochs_axis, costs, color="crimson", linewidth=2.5, marker="o", markersize=4)
    plt.title(f"1. Tien trinh giam Loss - {dataset_name.upper()} (n_h={n_h})", fontsize=12, fontweight="bold")
    plt.xlabel("So epoch", fontsize=10)
    plt.ylabel("Cost (Binary Cross-Entropy)", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)

    # 2. Decision boundary
    plt.subplot(1, 2, 2)
    x_min, x_max = X[0, :].min() - 0.5, X[0, :].max() + 0.5
    y_min, y_max = X[1, :].min() - 0.5, X[1, :].max() + 0.5
    h_step = 0.02
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h_step), np.arange(y_min, y_max, h_step))
    
    grid_points = np.c_[xx.ravel(), yy.ravel()].T
    Z_preds, _ = predict(params, grid_points)
    Z_preds = Z_preds.reshape(xx.shape)

    plt.contourf(xx, yy, Z_preds, cmap="Spectral", alpha=0.45)
    plt.scatter(X[0, :], X[1, :], c=Y.ravel(), s=35, cmap="Spectral", edgecolors="k", linewidth=0.5)
    plt.title(f"2. Ranh gioi phan lop - {label_vn} (Acc: {accuracy:.1f}%)", fontsize=12, fontweight="bold")
    plt.xlabel("X1", fontsize=10)
    plt.ylabel("X2", fontsize=10)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    plt.tight_layout()
    output_image = os.path.join(CURRENT_DIR, f"result_{dataset_name}.png")
    plt.savefig(output_image, dpi=160)
    plt.close()
    print(f"[+] Da luu anh truc quan hoa vao: {output_image}")
