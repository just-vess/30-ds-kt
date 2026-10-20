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

# Thu muc hien tai cua script de luu anh
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# ==========================================
# 1. CAC HAM COT LOI (FROM SCRATCH)
# ==========================================

def sigmoid(z):
    """
    Ham kich hoat Sigmoid: sigma(z) = 1 / (1 + exp(-z))
    Co clip de chong tran so (numerical overflow).
    """
    z_clip = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z_clip))

def parameter(dim):
    """
    Khoi tao trong so w shape (dim, 1) toan so 0 va b = 0.0
    """
    w = np.zeros((dim, 1))
    b = float(0.0)
    return w, b

def propagate(w, b, X, Y):
    """
    Forward pass: tinh gia tri du doan A va Binary Cross-Entropy Loss
    Backward pass: tinh dao ham dw, db bang vectorization
    """
    m = X.shape[1]
    
    # 1. Forward propagation
    Z = np.dot(w.T, X) + b
    A = sigmoid(Z)

    # Binary Cross-Entropy Loss (them eps tranh log(0))
    eps = 1e-15
    cost = -(1.0 / m) * np.sum(Y * np.log(A + eps) + (1.0 - Y) * np.log(1.0 - A + eps))

    # 2. Backward propagation
    dZ = A - Y
    dw = (1.0 / m) * np.dot(X, dZ.T)
    db = (1.0 / m) * np.sum(dZ)
    
    grads = {"dw": dw, "db": db}
    return grads, cost

# ==========================================
# 2. TOI UU HOA (GRADIENT DESCENT) & DU DOAN
# ==========================================

def optimize(w, b, X, Y, num_iterations=1000, learning_rate=0.1, print_cost=True):
    """
    Cap nhat w va b theo thuat toan Gradient Descent qua num_iterations epoch
    """
    costs = []
    
    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)
        
        dw = grads["dw"]
        db = grads["db"]
        
        # Cap nhat tham so
        w = w - learning_rate * dw
        b = b - learning_rate * db
        
        if i % 100 == 0:
            costs.append(cost)
            if print_cost:
                print(f"Epoch {i:4d}: Cost = {cost:.5f}")
                
    params = {"w": w, "b": b}
    return params, costs

def predict(w, b, X):
    """
    Du doan nhan 0 hoac 1 dua tren nguong xac suat 0.5
    """
    A = sigmoid(np.dot(w.T, X) + b)
    predictions = (A > 0.5).astype(int)
    return predictions, A

# ==========================================
# 3. THUC THI TEST & VISUALIZE
# ==========================================

if __name__ == "__main__":
    print("[*] Bat dau chay huan luyen va danh gia Single Neuron...")
    
    # 1. Tao du lieu mau 2D gom 2 cum tach biet
    np.random.seed(42)
    m = 200 # 200 diem
    
    # Lop 0 quanh (-1.5, -1.5)
    X0 = np.random.randn(2, m // 2) * 0.8 - 1.5
    Y0 = np.zeros((1, m // 2))
    
    # Lop 1 quanh (1.5, 1.5)
    X1 = np.random.randn(2, m // 2) * 0.8 + 1.5
    Y1 = np.ones((1, m // 2))
    
    X = np.hstack((X0, X1))
    Y = np.hstack((Y0, Y1))
    
    # 2. Khoi tao va huan luyen
    w, b = parameter(dim=2)
    
    # Cho phep truyen learning_rate tu dong lenh (vi du: python ... 0.01)
    if len(sys.argv) > 1:
        try:
            learning_rate = float(sys.argv[1])
            print(f"[i] Su dung Learning Rate truyen vao tu dong lenh: lr = {learning_rate}")
        except ValueError:
            learning_rate = 0.1
            print(f"[!] Khong the doc gia tri, su dung Learning Rate mac dinh: lr = {learning_rate}")
    else:
        learning_rate = 0.1
        print(f"[i] Su dung Learning Rate mac dinh: lr = {learning_rate}")

    num_iterations = 1000
    params, costs = optimize(w, b, X, Y, num_iterations=num_iterations, learning_rate=learning_rate)
    
    w_final = params["w"]
    b_final = params["b"]
    
    # 3. Danh gia do chinh xac
    y_pred, _ = predict(w_final, b_final, X)
    accuracy = np.mean(y_pred == Y) * 100
    print(f"\n[+] Ket qua huan luyen:")
    print(f"    w1 = {w_final[0,0]:.4f}, w2 = {w_final[1,0]:.4f}, b = {b_final:.4f}")
    print(f"    Do chinh xac (Accuracy): {accuracy:.2f}%\n")
    
    # ==========================================
    # TEST 1: TRUC QUAN HOA DUONG CONG LOSS
    # ==========================================
    plt.figure(figsize=(7, 5))
    epochs = np.arange(len(costs)) * 100
    plt.plot(epochs, costs, color="crimson", linewidth=2.5, marker="o", markersize=5)
    plt.title(f"Test 1: Learning Curve (lr = {learning_rate})", fontsize=12, fontweight="bold")
    plt.xlabel("Số vòng lặp (Epochs)", fontsize=10)
    plt.ylabel("Binary Cross-Entropy Cost", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    test1_path = os.path.join(CURRENT_DIR, "test1_loss_curve.png")
    plt.savefig(test1_path, dpi=150)
    plt.close()
    print(f"[+] Da luu Test 1 vao: {test1_path}")
    
    # ==========================================
    # TEST 2: TRUC QUAN HOA RANH GIOI QUYET DINH
    # ==========================================
    plt.figure(figsize=(7, 5))
    plt.scatter(X0[0, :], X0[1, :], color="red", label="Lớp 0 (Nhãn 0)", edgecolors="k", alpha=0.8, s=45)
    plt.scatter(X1[0, :], X1[1, :], color="blue", label="Lớp 1 (Nhãn 1)", edgecolors="k", alpha=0.8, s=45)
    
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid_points = np.c_[xx.ravel(), yy.ravel()].T
    _, grid_probs = predict(w_final, b_final, grid_points)
    grid_probs = grid_probs.reshape(xx.shape)
    
    plt.contourf(xx, yy, grid_probs, levels=20, cmap="coolwarm", alpha=0.25)
    
    # Duong ranh gioi p = 0.5: x2 = -(w1*x1 + b)/w2
    x_vals = np.array([x_min, x_max])
    decision_y = -(w_final[0, 0] * x_vals + b_final) / w_final[1, 0]
    plt.plot(x_vals, decision_y, color="black", linestyle="--", linewidth=2.5, label="Ranh giới quyết định (p = 0.5)")
    
    plt.title(f"Test 2: Decision Boundary (Độ chính xác: {accuracy:.1f}%)", fontsize=12, fontweight="bold")
    plt.xlabel("Đặc trưng X1", fontsize=10)
    plt.ylabel("Đặc trưng X2", fontsize=10)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    test2_path = os.path.join(CURRENT_DIR, "test2_decision_boundary.png")
    plt.savefig(test2_path, dpi=150)
    plt.close()
    print(f"[+] Da luu Test 2 vao: {test2_path}")
    
    # ==========================================
    # ANH TONG HOP: CA 2 TEST CANH NHAU
    # ==========================================
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, costs, color="crimson", linewidth=2.5, marker="o", markersize=4)
    plt.title(f"1. Đường cong giảm Loss (lr = {learning_rate})", fontsize=12, fontweight="bold")
    plt.xlabel("Epochs", fontsize=10)
    plt.ylabel("BCE Loss", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)
    
    plt.subplot(1, 2, 2)
    plt.scatter(X0[0, :], X0[1, :], color="red", label="Lớp 0", edgecolors="k", alpha=0.8, s=45)
    plt.scatter(X1[0, :], X1[1, :], color="blue", label="Lớp 1", edgecolors="k", alpha=0.8, s=45)
    plt.contourf(xx, yy, grid_probs, levels=20, cmap="coolwarm", alpha=0.25)
    plt.plot(x_vals, decision_y, color="black", linestyle="--", linewidth=2.5, label="Ranh giới (p=0.5)")
    plt.title(f"2. Ranh giới quyết định (lr = {learning_rate}, Acc: {accuracy:.1f}%)", fontsize=12, fontweight="bold")
    plt.xlabel("X1", fontsize=10)
    plt.ylabel("X2", fontsize=10)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.6)
    
    plt.tight_layout()
    summary_path = os.path.join(CURRENT_DIR, "so_sanh_tests.png")
    plt.savefig(summary_path, dpi=150)
    plt.close()
    print(f"[+] Da luu Anh tong hop vao: {summary_path}")
