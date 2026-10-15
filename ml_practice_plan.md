# 🎯 KẾ HOẠCH 30 NGÀY LOCK-IN: BÁM SÁT 100% SYLLABUS CỦA CÔ
## HƯỚNG 6: PHÁT TRIỂN CÁC PHƯƠNG PHÁP HỌC BIỂU DIỄN ĐA MÔ THỨC
> **Cố vấn hướng dẫn**: TS. Lê Hoàng Quỳnh · NCS. Cấn Duy Cát  
> **Lộ trình**: Level 1. CoreAI (Deep Learning Specialization + GenAI with LLMs) $\rightarrow$ Level 2. Track 3. Computer Vision (Unit 1 – Unit 5)  
> **Thời gian**: 30 ngày full lock-in · 3.5 – 4 tiếng/ngày  
> **Tài nguyên học**:
> - **Lý thuyết**: Toàn bộ slide, video Andrew Ng & giáo trình bạn đã up lên **NotebookLM** (tra cứu 45 phút đầu buổi).
> - **Bài tập Code**: Bám sát hệ thống bài tập thực hành mới nhất của **Viện IAI (UET - ĐHQGHN)**: *IAI Machine Learning (2526-2)* và *IAI Xử lý & Phân tích Hình ảnh (2526-2)*.
> - **Nguyên tắc Syllabus**: 
>   - Đã học xong: Linear Regression, Logistic Regression, Phân cụm (Clustering), Anomaly Detection.
>   - Bỏ qua toàn bộ phần bôi xám / gạch ngang / `SKIP` trong syllabus: *Heroes of Deep Learning*, *DLS C4 W2 (Case studies)*, *DLS C5 W3 (Sequence Models trùng GenAI)*, *GenAI W3 (RLHF)*, và các Track NLP / RAG không thuộc Hướng 6.

---

## ⏱️ Khung giờ thực hiện mỗi ngày (3.5 – 4 tiếng)

```text
┌─────────────────┬────────────────────────────────────────────────────────────────────────┐
│ Thời gian       │ Hoạt động chi tiết                                                     │
├─────────────────┼────────────────────────────────────────────────────────────────────────┤
│ 00:00 – 00:45   │ 🧠 Tra cứu lý thuyết trên NotebookLM (Công thức toán, Loss, Kiến trúc) │
│ 00:45 – 03:00   │ 💻 Code thực hành từ scratch / PyTorch / Bài tập IAI (Tự tay code)    │
│ 03:00 – 03:30   │ 📊 Trực quan hóa kết quả (Loss curve, Feature/Attention Map, Metric)    │
│ 03:30 – 03:45   │ 📝 Viết 3 dòng tổng kết kỹ thuật & Git Commit vào repo                 │
└─────────────────┴────────────────────────────────────────────────────────────────────────┘
```

---

## 🗺️ BẢNG TỔNG QUAN LỘ TRÌNH 30 NGÀY

| Giai đoạn | Nội dung trong Syllabus của Cô | Số ngày | Thời gian | Trọng tâm bài tập IAI & Deliverables |
| :--- | :--- | :---: | :---: | :--- |
| **Phần 1** | **DLS Course 1: Neural Networks and Deep Learning** | 3 ngày | Ngày 1 – 3 | Forward/Backpropagation từ scratch, Vectorization, Deep L-layer NN (IAI ML HW #11) |
| **Phần 2** | **DLS Course 2: Improving Deep Neural Networks** | 3 ngày | Ngày 4 – 6 | Regularization L2/Dropout (IAI ML HW #4), AdamW, Batch Normalization |
| **Phần 3** | **DLS Course 3: Structuring Machine Learning Projects** | 1 ngày | Ngày 7 | ML Strategy, Error Analysis, Confusion Matrix, Imbalanced Metric |
| **Phần 4** | **DLS Course 4: Convolutional Neural Networks (W1)** | 1 ngày | Ngày 8 | Khối tích chập 2D, Pooling, CNN Architecture (IAI CV Lec 9) |
| **Phần 5** | **DLS Course 5: Sequence Models (W1, W2)** | 2 ngày | Ngày 9 – 10 | RNN, LSTM, Word Embeddings (Word2Vec/GloVe) |
| **Phần 6** | **Generative AI with Large Language Models (W1, W2)** | 2 ngày | Ngày 11 – 12 | Transformer Architecture, Self-Attention, PEFT / LoRA |
| **Phần 7** | **Level 2 Track 3: Unit 1 - Fundamentals** | 2 ngày | Ngày 13 – 14 | Pre-processing, CLAHE, Filtering, Canny, HOG, SIFT, RANSAC (IAI CV Lec 1, 2, 4, 8) |
| **Phần 8** | **Level 2 Track 3: Unit 2 - Convolutional Neural Networks** | 3 ngày | Ngày 15 – 17 | VGG, GoogLeNet, ResNet, MobileNet, ConvNeXt, YOLO Detection (IAI CV Lec 10, 11) |
| **Phần 9** | **Level 2 Track 3: Unit 3 - Vision Transformers** | 4 ngày | Ngày 18 – 21 | ViT from scratch, Swin, DETR Object Detection, Segmentation U-Net/OneFormer (IAI CV Lec 10, 11, 12) |
| **Phần 10** | **Level 2 Track 3: Unit 4 - Multimodal Models (LÕI HƯỚNG 6)** | 4 ngày | Ngày 22 – 25 | CLIP from scratch, Symmetric InfoNCE Loss, Zero-shot, BLIP, OWL-ViT (IAI CV Lec 15) |
| **Phần 11** | **Level 2 Track 3: Unit 5 - Generative Models** | 3 ngày | Ngày 26 – 28 | VAE (IAI ML HW #13), GAN, CycleGAN, DDPM Diffusion Core (IAI CV Lec 14) |
| **Phần 12** | **Capstone Project & Báo cáo Onboard Lab** | 2 ngày | Ngày 29 – 30 | Multimodal Representation Learning trên dữ liệu y sinh + Hoàn thiện GitHub Repo |

---

## 📅 CHI TIẾT TỪNG NGÀY THEO CHUẨN SYLLABUS

---

### 🔵 LEVEL 1. CORE AI

#### 🔹 Deep Learning Specialization — Course 1: Neural Networks and Deep Learning [3 ngày]

* **Ngày 1 · Week 1 & Week 2: Introduction to Deep Learning & Neural Networks Basics**
  - **Syllabus của Cô**:
    - *Week 1*: Welcome to Deep Learning, Introduction to Deep Learning.
    - *Week 2*: Logistic Regression as a Neural Network, Python and Vectorization. *(Đã bỏ: Heroes of DL)*.
  - **Lý thuyết (NotebookLM)**: Biểu diễn mạng nơ-ron dưới dạng tính toán đồ thị (Computation Graph), Cơ chế Vectorization ma trận không dùng vòng lặp, Hàm mất mát Binary Cross-Entropy.
  - **Bài tập IAI**: **IAI ML Bài tập #11** (*Bài 1: Lựa chọn Loss Function; Bài 3: Forward Propagation*).
  - **Nhiệm vụ Code**: Viết thuật toán Logistic Regression dưới dạng 1 Single Neuron hoàn toàn vector hóa bằng NumPy: `z = np.dot(w.T, X) + b`, tính activation sigmoid, tính loss và gradient.
  - **Deliverable**: `day01_dls_c1_single_neuron.py`.

* **Ngày 2 · Week 3: Shallow Neural Networks**
  - **Syllabus của Cô**:
    - *Week 3*: Shallow Neural Network (Computing Neural Network Output, Vectorizing across multiple examples, Activation functions, Derivatives of activation functions, Gradient descent for Neural Networks). *(Đã bỏ: Heroes of DL)*.
  - **Lý thuyết (NotebookLM)**: Đạo hàm hàm kích hoạt (ReLU, LeakyReLU, Tanh, Sigmoid), Lan truyền ngược tầng ẩn (Chain Rule), Khởi tạo trọng số ngẫu nhiên (Random initialization vs Zero initialization).
  - **Bài tập IAI**: **IAI ML Bài tập #11** (*Bài 2: Hiện tượng bùng nổ / triệt tiêu gradient; Bài 4: Lan truyền ngược Backpropagation*).
  - **Nhiệm vụ Code**: Tự viết mạng 2 tầng (1 hidden layer + 1 output layer) bằng NumPy: hàm `forward()`, `backward()`, cập nhật trọng số và huấn luyện trên dữ liệu hình xoắn ốc (Planar data).
  - **Deliverable**: `day02_dls_c1_shallow_nn.py`.

* **Ngày 3 · Week 4: Deep Neural Networks**
  - **Syllabus của Cô**:
    - *Week 4*: Deep Neural Network (Deep L-layer neural network, Forward and Backward Propagation in a deep network, Parameters vs Hyperparameters).
  - **Lý thuyết (NotebookLM)**: Kiến trúc tổng quát $L$ tầng ẩn, Lưu trữ Cache kích hoạt $Z^{[l]}, A^{[l]}$ trong Forward để dùng cho Backward, Kích thước Tensor qua từng tầng $(n^{[l]}, n^{[l-1]})$.
  - **Nhiệm vụ Code**: 
    - Lắp ráp mô hình $L$-layer Deep Neural Network tổng quát bằng Python/NumPy: cho phép truyền vào cấu hình tầng bất kỳ (ví dụ: `[input_dim, 64, 32, 16, 1]`).
    - Chuyển đổi mã sang **PyTorch** sử dụng `torch.nn.Module` và xây dựng Custom Training Loop đầu tiên.
  - **Deliverable**: `day03_dls_c1_deep_l_layer_nn.py`.

---

#### 🔹 Deep Learning Specialization — Course 2: Improving Deep Neural Networks [3 ngày]

* **Ngày 4 · Week 1: Practical Aspects of Deep Learning**
  - **Syllabus của Cô**:
    - *Week 1*: Setting up your Machine Learning Application (Train/Dev/Test sets, Bias/Variance), Regularizing your Neural Network (L2 Regularization, Dropout), Setting Up your Optimization Problem (Normalizing inputs, Vanishing/Exploding gradients, Gradient Checking). *(Đã bỏ: Heroes of DL)*.
  - **Lý thuyết (NotebookLM)**: Trade-off Bias - Variance, Penalty $L_2$ (Weight Decay: $\frac{\lambda}{2m} \|W\|_F^2$), Inverted Dropout, Gradient Checking ($\epsilon = 10^{-7}$).
  - **Bài tập IAI**: **IAI ML Bài tập #4** (*Kiểm định chéo, Lựa chọn mô hình, Đánh giá Trade-off Bias/Variance*).
  - **Nhiệm vụ Code**:
    - Bổ sung $L_2$ Regularization và Inverted Dropout vào mạng nơ-ron.
    - Viết hàm kiểm tra đạo hàm số học (Gradient Checking) để debug hàm backward.
    - Vẽ đường cong Train Loss vs Val Loss khi thay đổi hệ số $\lambda$.
  - **Deliverable**: `day04_dls_c2_regularization_dropout.py`.

* **Ngày 5 · Week 2: Optimization Algorithms**
  - **Syllabus của Cô**:
    - *Week 2*: Optimization Algorithms (Mini-batch gradient descent, Exponentially weighted averages, Gradient descent with Momentum, RMSprop, Adam, Learning rate decay). *(Đã bỏ: Heroes of DL)*.
  - **Lý thuyết (NotebookLM)**: Trung bình động lũy thừa (EMA), Động lượng (Momentum $\beta_1$), RMSprop ($\beta_2$), Adam (kết hợp cả hai + hiệu chỉnh sai lệch Bias Correction), Lịch trình giảm tốc độ học (Learning Rate Schedules).
  - **Nhiệm vụ Code**:
    - Tự code 3 thuật toán tối ưu từ scratch: `SGD_Momentum`, `RMSprop`, `Adam`.
    - Huấn luyện cùng 1 bài toán trên PyTorch; vẽ đồ thị so sánh quỹ đạo hội tụ (Loss convergence path) của 3 thuật toán.
  - **Deliverable**: `day05_dls_c2_optimizers_adam.py`.

* **Ngày 6 · Week 3: Hyperparameter Tuning, Batch Normalization and Programming Frameworks**
  - **Syllabus của Cô**:
    - *Week 3*: Hyperparameter Tuning (Tuning process, Picking hyperparameters), Batch Normalization (Normalizing activations in a network, Fitting Batch Norm into a network), Multi-class Classification (Softmax Regression).
  - **Lý thuyết (NotebookLM)**: Cơ chế chuẩn hóa theo lô Batch Normalization (công thức $\mu_B, \sigma_B^2, \hat{z}, \gamma, \beta$), Cập nhật Running Mean & Running Variance trong lúc test, Hàm Softmax cho bài toán $C$ lớp.
  - **Bài tập IAI**: **IAI ML Bài tập #9** (*Bài 2: Phân lớp nhiều lớp One-vs-One vs One-vs-All vs Softmax*).
  - **Nhiệm vụ Code**:
    - Cài đặt thủ công lớp `CustomBatchNorm1d` và so sánh đầu ra với `torch.nn.BatchNorm1d`.
    - Tích hợp Softmax Cross-Entropy Loss cho bài toán phân loại đa lớp.
  - **Deliverable**: `day06_dls_c2_batchnorm_multiclass.py`.

---

#### 🔹 Deep Learning Specialization — Course 3: Structuring Machine Learning Projects [1 ngày]

* **Ngày 7 · Week 1 & Week 2: ML Strategy & Error Analysis**
  - **Syllabus của Cô**:
    - *Week 1*: Introduction to ML Strategy, Setting Up your Goal, Comparing to Human-level Performance.
    - *Week 2*: Error Analysis, Mismatched Training and Dev/Test Set, Learning from Multiple Tasks, End-to-end Deep Learning.
  - **Lý thuyết (NotebookLM)**: Chiến lược trực giao (Orthogonalization), Thiết lập Single-number Evaluation Metric (F1-score, Cohen Kappa, ROC-AUC cho y tế), Phân tích lỗi (Error Analysis bảng 100 mẫu nhầm lẫn), Xử lý phân phối dữ liệu lệch giữa Train và Dev/Test (Data Mismatch).
  - **Nhiệm vụ Code**:
    - Xây dựng một module Error Analysis toàn diện trên PyTorch: tự động trích xuất các mẫu bị phân loại sai nhiều nhất (Highest Loss misclassified samples).
    - Tính toán Ma trận nhầm lẫn (Confusion Matrix), Sensitivity, Specificity, F1-score và ROC-AUC cho dữ liệu mất cân bằng.
  - **Deliverable**: `day07_dls_c3_error_analysis.py`.

---

#### 🔹 Deep Learning Specialization — Course 4: Convolutional Neural Networks [1 ngày]

* **Ngày 8 · Week 1: Foundations of Convolutional Neural Networks**
  - **Syllabus của Cô**:
    - *Week 1*: Computer Vision, Edge Detection, Strided Convolutions, Convolutions Over Volume, One Layer of a Convolutional Network, Simple Convolutional Network Example, Pooling Layers (Max and Average Pooling). *(Đã bỏ: Week 2 Case Studies - SKIP)*.
  - **Lý thuyết (NotebookLM)**: Phép tính tích chập 3D $(H, W, C)$, Padding ('valid' vs 'same': $p = \frac{f-1}{2}$), Bước nhảy Stride, Kích thước đầu ra $\lfloor\frac{n+2p-f}{s}+1\rfloor$, Pooling không có tham số học.
  - **Bài tập IAI**: **IAI CV Lecture 9** (*Phân tích ảnh với CNN, Mạng nơ-ron tích chập*).
  - **Nhiệm vụ Code**:
    - Tự code hàm tích chập 2D đa kênh (`conv2d_forward`) bằng NumPy để hiểu sâu từng phép nhân ma trận chập.
    - Xây dựng mạng CNN chuẩn bằng PyTorch: `Conv2d` $\rightarrow$ `BatchNorm2d` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool2d` $\rightarrow$ `Linear`.
  - **Deliverable**: `day08_dls_c4_cnn_foundations.py`.

---

#### 🔹 Deep Learning Specialization — Course 5: Sequence Models [2 ngày]

* **Ngày 9 · Week 1: Recurrent Neural Networks**
  - **Syllabus của Cô**:
    - *Week 1*: Why sequence models, Notation, Recurrent Neural Network Model, Backpropagation through time, Different types of RNNs, Language model and sequence generation, Vanishing gradients with RNNs, Gated Recurrent Unit (GRU), Long Short Term Memory (LSTM).
  - **Lý thuyết (NotebookLM)**: Hidden state $h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$, Khối LSTM: Forget gate $f_t$, Input gate $i_t$, Candidate $\tilde{c}_t$, Cell state $c_t$, Output gate $o_t$.
  - **Nhiệm vụ Code**: Xây dựng mô hình Bi-directional LSTM bằng PyTorch (`nn.LSTM(bidirectional=True)`) xử lý dữ liệu chuỗi thời gian / tín hiệu y sinh.
  - **Deliverable**: `day09_dls_c5_rnn_lstm.py`.

* **Ngày 10 · Week 2: Natural Language Processing & Word Embeddings**
  - **Syllabus của Cô**:
    - *Week 2*: Word Representation, Using word embeddings, Properties of word embeddings, Cosine similarity, Embedding matrix, Learning Word Embeddings: Word2Vec (Skip-Gram & CBOW), Negative Sampling, GloVe. *(Đã bỏ: Week 3 Sequence Models - SKIP do trùng GenAI)*.
  - **Lý thuyết (NotebookLM)**: One-hot vector vs Dense Embedding, Ma trận tương đồng Cosine $\frac{u \cdot v}{\|u\| \|v\|}$, Bản chất Word2Vec Negative Sampling.
  - **Nhiệm vụ Code**:
    - Cài đặt mô hình Word2Vec (Skip-Gram with Negative Sampling) thu nhỏ bằng PyTorch.
    - Trích xuất ma trận Embeddings, tính khoảng cách Cosine giữa các thuật ngữ y khoa/từ vựng và trực quan hóa bằng PCA.
  - **Deliverable**: `day10_dls_c5_word_embeddings.py`.

---

#### 🔹 Generative AI with Large Language Models (DeepLearning.AI) [2 ngày]

* **Ngày 11 · Week 1: Generative AI use cases, project lifecycle, and model pre-training**
  - **Syllabus của Cô**:
    - *Week 1*: Transformers architecture, Transformers: Attention is all you need, Generating text with transformers, Multi-Head Attention, Scaled Dot-Product Attention, Positional Encoding, Pre-training large language models.
  - **Lý thuyết (NotebookLM)**: Công thức Attention cốt lõi: $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$, Multi-Head Attention, Sinusoidal Positional Encoding, Residual Connections & LayerNorm trong khối Transformer.
  - **Nhiệm vụ Code**: Tự viết toàn bộ khối `MultiHeadAttention` và `TransformerEncoderLayer` từ đầu bằng PyTorch (tự code phép nhân $Q, K, V$, scaling, softmax, mask).
  - **Deliverable**: `day11_genai_transformer_architecture.py`.

* **Ngày 12 · Week 2: Fine-tuning and evaluating large language models**
  - **Syllabus của Cô**:
    - *Week 2*: Instruction fine-tuning, Multi-task instruction fine-tuning, Model evaluation, Parameter-efficient fine-tuning (PEFT), PEFT techniques: LoRA (Low-Rank Adaptation), Soft prompts. *(Đã bỏ: Week 3 RLHF - SKIP)*.
  - **Lý thuyết (NotebookLM)**: Động lực của PEFT (giảm bộ nhớ VRAM khi train model lớn), Toán học của LoRA: phân rã ma trận cập nhật trọng số $\Delta W = B \cdot A$ với $B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times k}$ ($r \ll d$).
  - **Nhiệm vụ Code**: 
    - Tự code 1 lớp `LoRALinear(nn.Module)` thay thế cho `nn.Linear` để hiểu nguyên lý đóng băng $W_0$ và chỉ huấn luyện $A, B$.
    - Thực hành sử dụng thư viện `peft` của HuggingFace để áp dụng LoRA lên một Transformer backbone.
  - **Deliverable**: `day12_genai_peft_lora.py`.

---

### 🟢 LEVEL 2. TRACK 3. COMPUTER VISION (HƯỚNG 6)

#### 🔹 Unit 1 - Fundamentals [2 ngày]

* **Ngày 13 · Unit 1: Vision, Image, Imaging & Pre-processing for CV Tasks**
  - **Syllabus của Cô**:
    - *Unit 1*: Vision, Image, Imaging, Imaging in Real-life, What Is Computer Vision, Applications of Computer Vision, Pre-processing for Computer Vision Tasks.
  - **Lý thuyết (NotebookLM)**: Biểu diễn Tensor ảnh $(H, W, C)$, Lấy mẫu và lượng tử hóa, Biến đổi Gamma, Histogram Equalization, Cân bằng thích nghi có giới hạn tương phản (CLAHE), Lọc không gian (Gaussian, Median, Bilateral).
  - **Bài tập IAI**: **IAI CV Lecture 1 & 2** (*Thao tác mức điểm ảnh, Cân bằng biểu đồ mức xám, Lọc ảnh*).
  - **Nhiệm vụ Code**:
    - Cài đặt thuật toán CLAHE từ đầu bằng OpenCV/NumPy và áp dụng tăng tương phản ảnh X-quang/ảnh y tế độ tương phản kém.
    - Cài đặt Bilateral Filter để khử nhiễu ảnh mà vẫn giữ nguyên độ sắc nét của đường biên tổn thương.
  - **Deliverable**: `day13_cv_unit1_preprocessing_clahe.py`.

* **Ngày 14 · Unit 1: Feature Description, Feature Matching & Applications**
  - **Syllabus của Cô**:
    - *Unit 1*: Feature Description, Feature Matching, Real-world Applications of Feature Extraction in Computer Vision.
  - **Lý thuyết (NotebookLM)**: Điểm đặc trưng (Keypoints), Gradient Sobel, Canny Edge Detector, HOG (Histogram of Oriented Gradients), SIFT/ORB Descriptors, RANSAC loại trừ điểm ghép sai.
  - **Bài tập IAI**: **IAI CV Lecture 4 & 8** (*Đặc trưng ảnh: Cạnh, Canny; Đặc trưng cục bộ: HOG, SIFT, RANSAC, So khớp đặc trưng*).
  - **Nhiệm vụ Code**:
    - Trích xuất đặc trưng HOG của ảnh.
    - Sử dụng SIFT/ORB kết hợp FLANN Matcher và thuật toán RANSAC để so khớp tự động 2 bức ảnh y tế chụp ở các góc lệch nhau.
  - **Deliverable**: `day14_cv_unit1_features_ransac.py`.

---

#### 🔹 Unit 2 - Convolutional Neural Networks [3 ngày]

* **Ngày 15 · Unit 2: Classic CNN Architectures, ResNet & Transfer Learning**
  - **Syllabus của Cô**:
    - *Unit 2*: Introduction to Convolutional Neural Networks, Very Deep Convolutional Networks for Large Scale Image Recognition (VGG), GoogLeNet, ResNet, Introduction to Transfer Learning.
  - **Lý thuyết (NotebookLM)**: Kiến trúc VGG (đồng nhất filter $3 \times 3$), Inception Module (GoogLeNet), Vấn đề suy thoái (Degradation Problem) và giải pháp Residual Connection (Skip connection $F(x) + x$) trong ResNet.
  - **Bài tập IAI**: **IAI CV Lecture 10** (*Kiến trúc kinh điển: VGG, GoogLeNet, ResNet; Học chuyển giao*).
  - **Nhiệm vụ Code**:
    - Tự code class `ResidualBlock` và xây dựng ResNet-18 hoàn chỉnh bằng PyTorch.
    - Thực hiện Transfer Learning: Tải pre-trained weights từ `torchvision`, đóng băng backbone, thay thế classification head và fine-tune trên dữ liệu ảnh tế bào / MedMNIST.
  - **Deliverable**: `day15_cv_unit2_resnet_transfer.py`.

* **Ngày 16 · Unit 2: Efficient & Modern CNNs: MobileNet & ConvNeXt**
  - **Syllabus của Cô**:
    - *Unit 2*: MobileNet, ConvNeXt - A ConvNet for the 2020s, Let's Dive Further with MobileNet.
  - **Lý thuyết (NotebookLM)**: Depthwise Separable Convolution trong MobileNet (giảm chi phí tính toán $\approx \frac{1}{N} + \frac{1}{D_k^2}$), Inverted Residual Block (MobileNetV2), Triết lý thiết kế của ConvNeXt (áp dụng tư duy Transformer: 7x7 depthwise conv, inverted bottleneck, GELU, LayerNorm thay BatchNorm).
  - **Bài tập IAI**: **IAI CV Lecture 10** (*Kiến trúc hiện đại: ConvNeXt, EfficientNet*).
  - **Nhiệm vụ Code**:
    - Tự code 1 khối `DepthwiseSeparableConv` và 1 khối `ConvNeXtBlock` bằng PyTorch.
    - So sánh số lượng tham số (Parameters) và số phép tính (FLOPs) của ResNet vs MobileNet vs ConvNeXt.
  - **Deliverable**: `day16_cv_unit2_convnext_mobilenet.py`.

* **Ngày 17 · Unit 2: Object Detection Fundamentals & YOLO**
  - **Syllabus của Cô**:
    - *Unit 2*: YOLO: You Only Look Once.
  - **Lý thuyết (NotebookLM)**: Kiến trúc One-stage Object Detection, Bounding Box Regression $(x, y, w, h)$, Objectness Score, Intersection over Union (IoU), Non-Maximum Suppression (NMS).
  - **Bài tập IAI**: **IAI CV Lecture 11** (*Phát hiện đối tượng: Hai giai đoạn vs Một giai đoạn, YOLO, IoU, NMS*).
  - **Nhiệm vụ Code**:
    - Tự code hàm tính ma trận `IoU` và thuật toán `NMS` từ scratch bằng PyTorch Tensor.
    - Chạy inference với mô hình YOLO (Ultralytics YOLO) trên ảnh thực tế, trích xuất bounding boxes và độ tin cậy confidence.
  - **Deliverable**: `day17_cv_unit2_yolo_detection.py`.

---

#### 🔹 Unit 3 - Vision Transformers [4 ngày]

* **Ngày 18 · Unit 3: Vision Transformers (ViT) for Image Classification**
  - **Syllabus của Cô**:
    - *Unit 3*: Vision Transformers for Image Classification, Knowledge Distillation with Vision Transformers.
  - **Lý thuyết (NotebookLM)**: Chuyển đổi ảnh thành chuỗi token: Patch Extraction ($P \times P$), Linear Projection $E$, Learnable Position Embedding $E_{pos}$, Token phân loại `[CLS]`, Inductive Bias của CNN vs Khả năng tổng quát của ViT trên dữ liệu lớn.
  - **Bài tập IAI**: **IAI CV Lecture 10** (*Bộ biến đổi thị giác - ViT, Học chuyển giao và tinh chỉnh*).
  - **Nhiệm vụ Code**:
    - Tự viết toàn bộ mạng Vision Transformer (ViT) từ đầu bằng PyTorch: `PatchEmbed`, `Attention`, `Block`, `VisionTransformer`.
    - Kiểm tra kích thước tensor qua từng lớp đảm bảo đúng chuẩn: $(B, C, H, W) \rightarrow (B, N, D) \rightarrow (B, D) \rightarrow (B, \text{num\_classes})$.
  - **Deliverable**: `day18_cv_unit3_vit_implementation.py`.

* **Ngày 19 · Unit 3: Advanced ViT Variants (Swin, CvT, MobileViT)**
  - **Syllabus của Cô**:
    - *Unit 3*: Swin Transformer, Convolutional Vision Transformer (CvT), Dilated Neighborhood Attention Transformer (DINAT), MobileViT v2.
  - **Lý thuyết (NotebookLM)**: Hạn chế tính toán bậc hai của ViT $O((HW)^2)$, Cơ chế Shifted Windows (Swin Transformer) phân cấp độ phân giải và tính toán cục bộ đạt độ phức tạp tuyến tính $O(HW)$, Sự kết hợp Conv + Transformer trong MobileViT.
  - **Nhiệm vụ Code**:
    - Khởi tạo Swin Transformer từ thư viện `timm` (`timm.create_model('swin_tiny_patch4_window7_224', pretrained=True)`).
    - Viết thuật toán Attention Rollout để trực quan hóa bản đồ phân bổ chú ý (Attention Map) của mô hình Transformer trên ảnh.
  - **Deliverable**: `day19_cv_unit3_swin_transformer.py`.

* **Ngày 20 · Unit 3: Transformers for Object Detection (DETR)**
  - **Syllabus của Cô**:
    - *Unit 3*: Fine-tuning Vision Transformer for Object Detection, DEtection TRansformer (DETR).
  - **Lý thuyết (NotebookLM)**: Kiến trúc DETR (CNN Backbone + Transformer Encoder-Decoder + Object Queries), Không cần Anchor Box và không cần NMS, Thuật toán so khớp hai phía Hungarian Bipartite Matching Loss.
  - **Bài tập IAI**: **IAI CV Lecture 11** (*Dựa trên bộ biến đổi: DETR*).
  - **Nhiệm vụ Code**:
    - Tải mô hình pre-trained DETR (`facebook/detr-resnet-50`) từ HuggingFace.
    - Thực hiện forward pass, lọc các Object Queries có độ tin cậy cao, vẽ bounding boxes và so sánh cách hoạt động của DETR với YOLO.
  - **Deliverable**: `day20_cv_unit3_detr_detection.py`.

* **Ngày 21 · Unit 3: Transformers for Image Segmentation (U-Net & OneFormer)**
  - **Syllabus của Cô**:
    - *Unit 3*: Vision Transformers for Image Segmentation, OneFormer.
  - **Lý thuyết (NotebookLM)**: Phân vùng ngữ nghĩa (Semantic) vs Phân vùng thực thể (Instance) vs Phân vùng toàn cảnh (Panoptic), Kiến trúc hợp nhất OneFormer, U-Net Skip connections và Dice Loss.
  - **Bài tập IAI**: **IAI CV Lecture 12** (*Phân vùng ngữ nghĩa: FCN, U-Net; Phân vùng toàn cảnh: OneFormer*).
  - **Nhiệm vụ Code**:
    - Tự code kiến trúc mạng **U-Net** chuẩn bằng PyTorch: `DoubleConv`, `Down`, `Up`, `OutConv`.
    - Cài đặt hàm mất mát **Dice Loss** kết hợp **BCE Loss** ($\mathcal{L} = \text{BCE} + \text{Dice}$) để huấn luyện phân vùng tổn thương/vùng ảnh y tế.
  - **Deliverable**: `day21_cv_unit3_segmentation_unet.py`.

---

#### 🔹 Unit 4 - Multimodal Models (TRỌNG TÂM HƯỚNG 6) [4 ngày]

* **Ngày 22 · Unit 4: Introduction to Vision Language Models & Multimodal Tasks**
  - **Syllabus của Cô**:
    - *Unit 4*: Exploring Multimodal Text and Vision Models, A Multimodal World, Introduction to Vision Language Models, Multimodal Tasks and Models.
  - **Lý thuyết (NotebookLM)**: Bức tranh học biểu diễn đa mô thức (Multimodal Representation Learning), Các tác vụ cốt lõi: Visual Question Answering (VQA), Image Captioning, Cross-Modal Retrieval (Text-to-Image / Image-to-Text), Visual Grounding.
  - **Bài tập IAI**: **IAI CV Lecture 15** (*Mô hình ngôn ngữ thị giác và các tác vụ thị giác-ngôn ngữ, Kiến trúc hiện đại*).
  - **Nhiệm vụ Code**: Xây dựng pipeline kiểm thử các tác vụ đa mô thức với thư viện `transformers`: nạp cặp ảnh và văn bản, kiểm tra không gian đặc trưng đa mô thức.
  - **Deliverable**: `day22_cv_unit4_vlm_intro.py`.

* **Ngày 23 · Unit 4: CLIP & Contrastive Representation Learning**
  - **Syllabus của Cô**:
    - *Unit 4*: CLIP and Relatives, Losses, Contrastive Language-Image Pre-training (CLIP).
  - **Lý thuyết (NotebookLM)**: Kiến trúc Dual Encoders (Image Encoder $f_I$ + Text Encoder $f_T$), Phép chiếu vào không gian chung (Linear Projection), Hàm mất mát tương phản đối xứng **Symmetric InfoNCE Loss**:
    $$\mathcal{L} = \frac{1}{2} (\mathcal{L}_{I \rightarrow T} + \mathcal{L}_{T \rightarrow I}), \quad \mathcal{L}_{I \rightarrow T} = -\frac{1}{B}\sum_{i=1}^B \log \frac{\exp(I_i \cdot T_i / \tau)}{\sum_{j=1}^B \exp(I_i \cdot T_j / \tau)}$$
  - **Bài tập IAI**: **IAI CV Lecture 15** (*CLIP và học tương phản ảnh-văn bản*).
  - **Nhiệm vụ Code**:
    - Tự code toàn bộ mô hình `MiniCLIP` từ đầu bằng PyTorch (gồm Image Encoder, Text Encoder và Projection Heads).
    - Cài đặt hàm mất mát Symmetric InfoNCE Loss với tham số nhiệt độ $\tau$ có thể học (`nn.Parameter`).
  - **Deliverable**: `day23_cv_unit4_clip_from_scratch.py`.

* **Ngày 24 · Unit 4: Multimodal Text Generation (BLIP) & Multimodal Object Detection (OWL-ViT)**
  - **Syllabus của Cô**:
    - *Unit 4*: Multimodal Text Generation (BLIP), Multimodal Object Detection (OWL-ViT).
  - **Lý thuyết (NotebookLM)**: Cơ chế Fusion Encoder với Cross-Attention trong BLIP (Query lấy từ Text, Key/Value lấy từ Image Patches), Mở rộng phát hiện đối tượng mở bằng văn bản (Open-Vocabulary Detection với OWL-ViT).
  - **Bài tập IAI**: **IAI CV Lecture 15** (*Bộ chiếu, Mô hình ngôn ngữ lớn, Neo nghĩa vào vùng ảnh - Visual Grounding*).
  - **Nhiệm vụ Code**:
    - Cài đặt module `CrossAttentionFusion(nn.Module)` để kết nối biểu diễn giữa token văn bản và patch hình ảnh.
    - Sử dụng pre-trained BLIP (`Salesforce/blip-image-captioning-base`) thực hiện sinh tự động câu mô tả (Image Captioning) cho ảnh.
  - **Deliverable**: `day24_cv_unit4_blip_cross_attention.py`.

* **Ngày 25 · Unit 4: Transfer Learning & Zero-shot Evaluation of Multimodal Models**
  - **Syllabus của Cô**:
    - *Unit 4*: Transfer Learning of Multimodal Models.
  - **Lý thuyết (NotebookLM)**: Prompt Engineering cho Thị giác (*"a photo of [LABEL]"*), Zero-shot Classification, Cross-modal Retrieval (Đánh giá bằng Recall@1, Recall@5, Mean Reciprocal Rank - MRR), Tinh chỉnh hiệu quả tham số (PEFT/LoRA) trên CLIP.
  - **Bài tập IAI**: **IAI CV Lecture 15** (*Giới hạn và đánh giá mô hình ngôn ngữ thị giác*).
  - **Nhiệm vụ Code**:
    - Tải mô hình CLIP pre-trained (`openai/clip-vit-base-patch32` hoặc BiomedCLIP).
    - Xây dựng hệ thống Zero-shot Image Classifier và Image-Text Search Engine; tính toán ma trận Recall@K.
  - **Deliverable**: `day25_cv_unit4_clip_zeroshot_retrieval.py`.

---

#### 🔹 Unit 5 - Generative Models [3 ngày]

* **Ngày 26 · Unit 5: Autoencoders & Variational Autoencoders (VAE)**
  - **Syllabus của Cô**:
    - *Unit 5*: Introduction, Variational Autoencoders.
  - **Lý thuyết (NotebookLM)**: Phán đoán vs Tạo sinh (Discriminative vs Generative), Không gian tiềm ẩn (Latent space), Phân phối chuẩn hóa Gaussian, Reparameterization Trick ($z = \mu + \sigma \odot \epsilon$), Hàm mất mát ELBO (Reconstruction Loss + KL Divergence).
  - **Bài tập IAI**: **IAI ML Bài tập #13** (*Bài 1: Phán đoán vs Sinh; Bài 3: VAE vs GAN vs Diffusion*) & **IAI CV Lecture 14** (*Mô hình tự mã hóa, VAE*).
  - **Nhiệm vụ Code**:
    - Cài đặt mạng VAE đầy đủ bằng PyTorch (Encoder $\rightarrow \mu, \log\sigma^2 \rightarrow$ Reparameterize $\rightarrow$ Decoder).
    - Huấn luyện mô hình tái tạo ảnh và thực hiện nội suy không gian tiềm ẩn (Latent Space Interpolation) để quan sát sự biến đổi biểu diễn mượt mà giữa 2 bức ảnh.
  - **Deliverable**: `day26_cv_unit5_vae_representation.py`.

* **Ngày 27 · Unit 5: GANs, StyleGAN & CycleGAN**
  - **Syllabus của Cô**:
    - *Unit 5*: Generative Adversarial Networks, StyleGAN Variants, CycleGAN Introduction.
  - **Lý thuyết (NotebookLM)**: Minimax Game của Generator & Discriminator ($\min_G \max_D V(D, G)$), Mode Collapse, Cycle-Consistency Loss trong CycleGAN ($G_{B\rightarrow A}(G_{A\rightarrow B}(x)) \approx x$) cho phép biến đổi miền ảnh không cần dữ liệu cặp (unpaired domain translation).
  - **Bài tập IAI**: **IAI CV Lecture 14** (*Mô hình sinh đối nghịch: DCGAN, Pix2Pix, CycleGAN*).
  - **Nhiệm vụ Code**:
    - Tự code 3 hàm mất mát cốt lõi của CycleGAN: Adversarial Loss, Cycle Consistency Loss, và Identity Loss.
    - Khảo sát ứng dụng CycleGAN trong chuyển đổi ảnh y sinh (mô phỏng biến đổi giữa các chế độ chụp MRI T1 $\leftrightarrow$ T2 hoặc CT $\leftrightarrow$ MRI).
  - **Deliverable**: `day27_cv_unit5_cyclegan_losses.py`.

* **Ngày 28 · Unit 5: Diffusion Models & Stable Diffusion**
  - **Syllabus của Cô**:
    - *Unit 5*: Introduction to Diffusion Models, Introduction to Stable Diffusion, Control over Diffusion Models, Privacy, Bias and Societal Concerns.
  - **Lý thuyết (NotebookLM)**: Quá trình thuận Forward Diffusion (thêm nhiễu Gauss theo lịch trình $\beta_t$), Quá trình nghịch Reverse Denoising, U-Net Denoiser dự đoán nhiễu $\epsilon_\theta(x_t, t)$, Khái niệm Latent Diffusion (Stable Diffusion nén ảnh vào latent của VAE trước khi khuếch tán để tiết kiệm tính toán).
  - **Bài tập IAI**: **IAI CV Lecture 14** (*Mô hình khuếch tán DDPM, DDIM, Khớp dòng - Flow Matching, Sinh có điều kiện: Stable Diffusion, CLIP*).
  - **Nhiệm vụ Code**:
    - Cài đặt công thức Closed-form của quá trình thêm nhiễu: $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$.
    - Viết training step cơ bản huấn luyện mạng U-Net nhỏ dự đoán nhiễu với hàm mất mát MSE.
  - **Deliverable**: `day28_cv_unit5_ddpm_diffusion_core.py`.

---

#### 🔹 CAPSTONE HƯỚNG 6 & ONBOARD LAB [2 ngày]

* **Ngày 29 · Capstone Project: Huấn luyện Biểu diễn Đa mô thức trên Dữ liệu Y sinh Thực tế**
  - **Mục tiêu**: Xây dựng một bài toán nghiên cứu hoàn chỉnh theo đúng đề tài của TS. Lê Hoàng Quỳnh & NCS. Cấn Duy Cát: *Học biểu diễn đa mô thức ghép nối Ảnh X-quang/Tế bào với Báo cáo lâm sàng (Biomedical Image-Report Alignment)*.
  - **Nhiệm vụ Code**:
    - Chuẩn bị pipeline cặp dữ liệu nhỏ (Image + Clinical Text Report).
    - Tích hợp Vision Backbone (ResNet/ViT) + Text Backbone (BioBERT/ClinicalBERT) qua Projection Heads.
    - Huấn luyện tinh chỉnh với InfoNCE Loss kết hợp LoRA; trích xuất không gian đặc trưng đa mô thức.
    - Đo lường độ chính xác Zero-shot Retrieval và trực quan hóa không gian đa mô thức bằng t-SNE trước và sau khi học.
  - **Deliverable**: `day29_capstone_multimodal_biomed.py`.

* **Ngày 30 · Tổng kết Milestone, Hoàn thiện Repo & Sẵn sàng Báo cáo Lab**
  - **Nhiệm vụ trọng tâm**:
    - Rà soát toàn bộ 29 scripts code trong repo `30-ds-kt`.
    - Hoàn thiện `README.md` của repo: Mục lục 30 ngày bám sát 100% syllabus của Cô, bảng liên kết code, tổng hợp biểu đồ trực quan hóa (Loss curves, Grad-CAM heatmaps, t-SNE representations, Attention Rollout).
    - Lập bản tóm tắt năng lực thực hành và các bài tập IAI đã giải quyết để tự tin gặp TS. Lê Hoàng Quỳnh và NCS. Cấn Duy Cát, chính thức bước vào đề tài nghiên cứu tại Lab!
  - **Deliverable**: `README.md` hoàn chỉnh + GitHub Repo chuẩn mực phong cách nghiên cứu chuyên nghiệp.

---

## 🛠️ Checklist Môi trường Cần thiết

Môi trường đã có sẵn Python 3.13 và PyTorch trong `.venv`. Cài đặt đầy đủ các thư viện phụ trợ:
```powershell
# Chạy trong PowerShell tại C:\Users\HELLO\Documents\DSKT
.\.venv\Scripts\pip.exe install opencv-python opencv-contrib-python
.\.venv\Scripts\pip.exe install timm transformers datasets peft
.\.venv\Scripts\pip.exe install scikit-learn matplotlib seaborn pypdf ultralytics
```
