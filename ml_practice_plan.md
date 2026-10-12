# 🎯 Kế hoạch 30 ngày luyện CODE: CoreAI → CV (Ưu tiên 1) → NLP (Ưu tiên 2)

> **Mục tiêu**: Chuẩn bị lock-in Lab sau 30 ngày.  
> **Định hướng nghiên cứu**: Multimodal Representation Learning (Biomedical) · Foundation Models · Sequential Decision Making (RL).  
> **Chiến lược co gọn**: 
> - **Level 1 (Core AI)**: 12 ngày (Nền tảng vững, implement scratch + PyTorch).
> - **Level 2 Track CV (Ưu tiên 1)**: 9 ngày (CNN, ResNet, Vision Transformer - ViT, CLIP đa mô thức).
> - **Level 2 Track NLP (Ưu tiên 2)**: 6 ngày (Embeddings, Seq2Seq, Transformer Text, BERT fine-tuning).
> - **RL & Tổng kết**: 3 ngày (MDP, Q-Learning/DQN cơ bản làm bàn đạp cho Sequential Decision Making).
> - *(Đã cắt bỏ: Track RAG, Siamese Networks, Autocorrect, các phần SKIP trong syllabus).*

---

## 🛠️ Chuẩn bị môi trường (30 phút Ngày 0)
Trong thư mục `C:\Users\HELLO\Documents\DSKT\30-ds-kt`:
```bash
pip install torch torchvision numpy pandas matplotlib scikit-learn
pip install transformers datasets sentencepiece timm
pip install gymnasium
```

---

## 📅 CHẶNG 1: LEVEL 1 — CORE AI CODING (Ngày 1 – 12)

### 🔵 Phần 1: Machine Learning từ Scratch (Ngày 1 – 4)
*Lý thuyết bạn đã nắm chắc + tính tay xong, giờ là lúc biến toán thành code Python/NumPy.*

- **Ngày 1 · Linear Regression từ Scratch**:
  - Tự code: Hàm mất mát MSE, Gradient Descent vector hóa (không loop qua samples).
  - So sánh trực tiếp trọng số $W, b$ tìm được với `sklearn.linear_model.LinearRegression`.
  - *Tài nguyên*: [Google MLCC Linear Regression](https://developers.google.com/machine-learning/crash-course/linear-regression).
  - *Deliverable*: `day01_linear_regression.py`.

- **Ngày 2 · Logistic Regression & Đánh giá Y sinh**:
  - Tự code: Sigmoid, Binary Cross-Entropy Loss, Gradient Descent.
  - Vẽ đường biên phân lớp (Decision Boundary) + Ma trận nhầm lẫn (Confusion Matrix), ROC-AUC.
  - Test trên tập dữ liệu y sinh thực tế: `sklearn.datasets.load_breast_cancer()`.
  - *Deliverable*: `day02_logistic_regression.py`.

- **Ngày 3 · Regularization (L1/L2) & K-Fold Cross Validation**:
  - Bổ sung $L_2$ Penalty (Weight Decay) vào hàm mất mát và gradient.
  - Tự viết hàm chia K-Fold Cross-Validation từ đầu bằng NumPy.
  - Vẽ biểu đồ thể hiện Trade-off Bias - Variance khi thay đổi $\lambda$.
  - *Deliverable*: `day03_regularization.py`.

- **Ngày 4 · Decision Trees & Tree Ensembles**:
  - Dùng Scikit-learn xây dựng Random Forest & Gradient Boosting (XGBoost).
  - Đánh giá Feature Importance trên bảng dữ liệu bệnh án.
  - *Deliverable*: `day04_tree_ensembles.py`.

---

### 🔴 Phần 2: Deep Learning với PyTorch (Ngày 5 – 12)
*Xương sống của mọi mô hình hiện đại.*

- **Ngày 5 · PyTorch Tensor, Autograd & Linear Model**:
  - Thao tác Tensor nâng cao: broadcasting, tensor slicing, `einsum`.
  - Tự tạo computational graph và kiểm tra đạo hàm tự động `loss.backward()`.
  - Viết lại Linear & Logistic Regression hoàn toàn bằng `torch.nn.Module`.
  - *Tài nguyên*: [D2L.ai Ch. 2 Preliminaries](https://d2l.ai/chapter_preliminaries/index.html).
  - *Deliverable*: `day05_pytorch_basics.py`.

- **Ngày 6 · Multilayer Perceptron (MLP) & Custom Training Loop**:
  - Xây dựng MLP với PyTorch: `nn.Linear`, `nn.ReLU`, `nn.Dropout`.
  - Tự viết Training Loop chuẩn chỉnh: Train/Val split, Batch DataLoader, Loss logging.
  - Huấn luyện trên Fashion-MNIST.
  - *Tài nguyên*: [D2L.ai Ch. 5 MLP](https://d2l.ai/chapter_multilayer-perceptrons/index.html).
  - *Deliverable*: `day06_mlp_training.py`.

- **Ngày 7 · Optimization Algorithms (Cốt lõi cho RL sau này)**:
  - So sánh thực nghiệm qua code: SGD, Momentum, RMSProp và Adam.
  - Quan sát hiện tượng Learning Rate decay và Scheduler (`StepLR`, `CosineAnnealingLR`).
  - *Tài nguyên*: [D2L.ai Ch. 12 Optimization](https://d2l.ai/chapter_optimization/index.html).
  - *Deliverable*: `day07_optimizers_experiment.py`.

- **Ngày 8 · Convolutional Neural Network (CNN) Cơ bản**:
  - Tự code phép tính Convolution 2D bằng NumPy để hiểu bản chất Kernel, Stride, Padding.
  - Dùng PyTorch xây dựng mạng LeNet-5 (`nn.Conv2d`, `nn.MaxPool2d`).
  - *Tài nguyên*: [D2L.ai Ch. 7 CNN](https://d2l.ai/chapter_convolutional-neural-networks/index.html).
  - *Deliverable*: `day08_cnn_lenet.py`.

- **Ngày 9 · Modern CNN Architectures (ResNet & Skip Connections)**:
  - Tự xây dựng khối `ResidualBlock` (ResNet-18) giải quyết vấn đề Vanishing Gradient.
  - Huấn luyện ResNet-18 trên tập ảnh phân loại tế bào hoặc CIFAR-10.
  - *Tài nguyên*: [D2L.ai Ch. 8 Modern CNN](https://d2l.ai/chapter_convolutional-modern/resnet.html).
  - *Deliverable*: `day09_resnet.py`.

- **Ngày 10 · Sequence Model: RNN & LSTM**:
  - Cấu trúc `nn.LSTM` xử lý dữ liệu chuỗi (chuỗi thời gian / tín hiệu y sinh ECG).
  - Xử lý vấn đề biến thiên độ dài chuỗi (`pack_padded_sequence`).
  - *Tài nguyên*: [D2L.ai Ch. 9 RNN](https://d2l.ai/chapter_recurrent-neural-networks/index.html).
  - *Deliverable*: `day10_lstm_sequential.py`.

- **Ngày 11 · Cơ chế Attention & Transformer Encoder**:
  - Code phép toán Scaled Dot-Product Attention: $\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$.
  - Lắp ráp Multi-Head Attention và Positional Encoding thành 1 khối Transformer Encoder.
  - *Tài nguyên*: [D2L.ai Ch. 11 Attention Mechanisms](https://d2l.ai/chapter_attention-mechanisms-and-transformers/index.html).
  - *Deliverable*: `day11_transformer_block.py`.

- **Ngày 12 · Transfer Learning & Foundation Model Fine-tuning**:
  - Tải pre-trained weights từ HuggingFace/Torchvision, đóng băng backbone (freeze), thay thế classification head.
  - Làm quen với kỹ thuật tham số hiệu quả PEFT / LoRA cơ bản.
  - *Deliverable*: `day12_transfer_learning.py`.

---

## 📅 CHẶNG 2: LEVEL 2 — CV TRACK (Ưu tiên 1) (Ngày 13 – 21)
*Trọng tâm của biểu diễn ảnh y sinh & Vision-Language Foundation Models.*

- **Ngày 13 · Tiền xử lý ảnh Y sinh & Data Augmentation**:
  - Dùng `torchvision.transforms` để chuẩn hóa ảnh y sinh (Intensity clipping, Rotation, Random Affine).
  - Xây dựng PyTorch Dataset pipeline tối ưu với `DataLoader(num_workers, pin_memory)`.
  - *Deliverable*: `day13_biomed_data_pipeline.py`.

- **Ngày 14 · Modern ConvNets: ConvNeXt & MobileNet**:
  - Khảo sát các kiến trúc CNN hiện đại cho ứng dụng nhẹ (MobileNet) và dạng block hiện đại (ConvNeXt).
  - Đánh giá FLOPS, số lượng tham số và tốc độ suy luận (Inference Latency).
  - *Deliverable*: `day14_convnext_mobilenet.py`.

- **Ngày 15 · Object Detection cơ bản (YOLO intuition)**:
  - Khái niệm Bounding Box, Intersection over Union (IoU), Non-Maximum Suppression (NMS).
  - Thực hành chạy inference và phân tích output của một mô hình YOLO pretrained trên ảnh y tế/vật thể.
  - *Deliverable*: `day15_detection_pipeline.py`.

- **Ngày 16 · Vision Transformers (ViT) Architecture**:
  - Bản chất ViT: Cắt ảnh thành các Patch 16x16, xem mỗi patch như 1 "word token".
  - Code lớp Patch Embedding và học các Positional Embeddings 2D.
  - Tải pre-trained ViT (`google/vit-base-patch16-224`) từ HuggingFace và fine-tune.
  - *Deliverable*: `day16_vit_implementation.py`.

- **Ngày 17 · Biến thể ViT nâng cao (Swin Transformer)**:
  - Hiểu cơ chế Shifted Windows (Swin Transformer) giải quyết bài toán độ phức tạp tính toán trên ảnh kích thước lớn.
  - So sánh hiệu năng của CNN thông thường vs ViT vs Swin Transformer.
  - *Deliverable*: `day17_swin_transformer.py`.

- **Ngày 18 · Multimodal Models: CLIP (Contrastive Language-Image Pre-training)**:
  - Cốt lõi của Multimodal: Dual Encoders (Image Encoder + Text Encoder).
  - Code hàm mất mát InfoNCE Loss (Contrastive Cross-Entropy) căn chỉnh không gian embedding giữa Ảnh và Văn bản.
  - *Deliverable*: `day18_clip_contrastive_loss.py`.

- **Ngày 19 · Thực hành Zero-shot Classification với CLIP**:
  - Sử dụng mô hình `openai/clip-vit-base-patch32` trên Hugging Face.
  - Thực hiện phân loại ảnh zero-shot bằng text prompt (ví dụ: *"a photo of normal chest X-ray"* vs *"a photo of pneumonia"*).
  - *Deliverable*: `day19_clip_zeroshot.py`.

- **Ngày 20 · Explainability cho Vision: Grad-CAM (Khả năng giải thích)**:
  - Code kỹ thuật Grad-CAM: dùng gradient của class score theo feature map cuối cùng để vẽ Heatmap vùng ảnh mô hình chú ý vào.
  - Cực kỳ quan trọng đối với dữ liệu y tế (Bác sĩ cần biết mô hình nhìn vào đâu để chẩn đoán).
  - *Deliverable*: `day20_gradcam_interpretable.py`.

- **Ngày 21 · Generative Models cơ bản: VAE (Variational Autoencoder)**:
  - Code cấu trúc Encoder - Latent Vector ($\mu, \sigma$) - Reparameterization Trick - Decoder.
  - Khám phá không gian tiềm ẩn (Latent space) của ảnh.
  - *Deliverable*: `day21_vae_representation.py`.

---

## 📅 CHẶNG 3: LEVEL 2 — NLP TRACK (Ưu tiên 2) (Ngày 22 – 27)
*Xử lý văn bản, Bệnh án điện tử (EHR/Clinical notes) & Foundation Model text encoder.*

- **Ngày 22 · Vector Space Models & Word Embeddings**:
  - Tính toán ma trận Cosine Similarity giữa các từ và văn bản.
  - Huấn luyện Word2Vec (CBOW) nhỏ hoặc trích xuất embedding từ mô hình ngôn ngữ pre-trained.
  - *Deliverable*: `day22_word_embeddings.py`.

- **Ngày 23 · Named Entity Recognition (NER) cho Y sinh**:
  - Xây dựng mô hình Bi-LSTM / Transformer phân loại thực thể y khoa (Tên thuốc, Triệu chứng, Liều lượng) từ câu văn.
  - *Deliverable*: `day23_biomed_ner.py`.

- **Ngày 24 · Pre-trained Language Models: BERT & Tokenization**:
  - Làm chủ HuggingFace Tokenizer (BPE/WordPiece, Attention Mask, Token Type IDs).
  - Sử dụng BERT / ClinicalBERT để trích xuất Feature Representation từ đoạn văn bản y khoa.
  - *Deliverable*: `day24_bert_feature_extractor.py`.

- **Ngày 25 · Fine-tuning Text Classification**:
  - Fine-tune BERT cho bài toán phân loại sắc thái hoặc phân loại bệnh lý từ đoạn ghi chú lâm sàng.
  - Đánh giá F1-score và phân tích lỗi (Error Analysis).
  - *Deliverable*: `day25_finetune_text_classifier.py`.

- **Ngày 26 · Cross-Attention: Nối Ảnh và Văn bản**:
  - Tự viết module Cross-Attention: Query lấy từ Text Tokens, Key/Value lấy từ Image Patch Tokens (cơ chế của BLIP/VLM).
  - Đây là viên gạch kết nối trực tiếp CV và NLP cho hướng Multimodal Foundation Models.
  - *Deliverable*: `day26_cross_attention_multimodal.py`.

- **Ngày 27 · Interpretability cho Text: Attention Rollout & Saliency**:
  - Trích xuất Attention Weights từ Transformer để xem các từ ngữ nào liên kết mạnh nhất với đặc trưng của ảnh.
  - *Deliverable*: `day27_text_interpretability.py`.

---

## 📅 CHẶNG 4: REINFORCEMENT LEARNING & HOÀN THIỆN (Ngày 28 – 30)
*Bàn đạp cho Sequential Decision Making (Ra quyết định tuần tự).*

- **Ngày 28 · Cơ sở RL: Markov Decision Process (MDP) & Q-Learning**:
  - Làm quen với môi trường [Gymnasium](https://gymnasium.farama.org/) (`CartPole` hoặc `FrozenLake`).
  - Code thuật toán Tabular Q-Learning: Cập nhật Bellman Equation.
  - *Deliverable*: `day28_tabular_qlearning.py`.

- **Ngày 29 · Deep Q-Network (DQN) - Nối Neural Network vào RL**:
  - Xây dựng Q-Network bằng PyTorch thay thế bảng Q-table.
  - Tự code Replay Buffer và Target Network để ổn định quá trình học ra quyết định tuần tự.
  - *Deliverable*: `day29_dqn_cartpole.py`.

- **Ngày 30 · Tổng kết Milestone & Dọn dẹp Repo**:
  - Review toàn bộ các script từ Ngày 1 đến Ngày 29.
  - Hoàn thiện `README.md` của repo `30-ds-kt` với bảng mục lục, kết quả biểu đồ đã vẽ, và danh mục code mẫu.
  - Sẵn sàng mang repo này gặp thầy cô và mentor khi bắt đầu giai đoạn lock-in!

---

## 📊 Phân bổ quỹ thời gian mỗi ngày (3 tiếng)

```
[00:00 - 00:30] Đọc lý thuyết / Xem công thức toán cần code
[00:30 - 02:30] Code trực tiếp trên VS Code / Jupyter Notebook (Tự gõ, không copy-paste)
[02:30 - 03:00] Debug, vẽ biểu đồ trực quan hóa, viết 2-3 dòng giải thích và Git Commit
```
