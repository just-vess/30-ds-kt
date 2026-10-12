# 🗓️ Kế hoạch 30 ngày — Code theo Syllabus Level 1 & 2

> **Đã có**: Lý thuyết + bài tập tính tay Week 1–3 ML Spec ✅  
> **Scope**: Code song song + code lại phần đã học · Level 1 (CoreAI) → Level 2 (NLP · RAG · CV)  
> **Budget**: 3h/ngày · **Không có ngày đọc lý thuyết** — chỉ code

> [!NOTE]
> Các phần **SKIP** trong syllabus được đánh dấu rõ — không mất thời gian vào đó.

---

## ⚡ Setup — Làm trước ngày 1 (30 phút)

```bash
pip install torch torchvision numpy pandas matplotlib scikit-learn
pip install gymnasium nltk transformers datasets
pip install sentence-transformers faiss-cpu
```
Tạo repo `lab-level1-2` trên GitHub — push code mỗi ngày.

---

## 📅 LEVEL 1 — CoreAI Coding (Ngày 1–15)

---

### 🔵 ML Specialization Coding (Ngày 1–5)

---

#### Ngày 1 · Code lại: Linear Regression từ scratch
*Bạn đã chứng minh tay → giờ implement*

| | |
|---|---|
| **Resource** | [Google MLCC - Linear Regression](https://developers.google.com/machine-learning/crash-course/linear-regression) |
| **Implement** | MSE loss · Gradient descent · Train loop · Learning rate schedule |
| **Verify** | So sánh với `sklearn.LinearRegression` — phải cho cùng kết quả |
| **Deliverable** | `day01_linear_regression.py` + loss curve plot |

---

#### Ngày 2 · Code lại: Logistic Regression từ scratch
*Week 3 ML Spec đã học lý thuyết*

| | |
|---|---|
| **Resource** | [Google MLCC - Logistic Regression](https://developers.google.com/machine-learning/crash-course/logistic-regression) |
| **Implement** | Sigmoid · Binary cross-entropy · Gradient descent · Decision boundary 2D |
| **Metrics** | Accuracy · Precision · Recall · F1 · ROC-AUC |
| **Deliverable** | `day02_logistic_regression.py` + ROC curve |

---

#### Ngày 3 · Code: Bias-Variance · Regularization (ML Spec Course 2 Week 3)

| | |
|---|---|
| **Resource** | [D2L Ch.3.6 Generalization](https://d2l.ai/chapter_linear-regression/generalization.html) + [Ch.3.7 Weight Decay](https://d2l.ai/chapter_linear-regression/weight-decay.html) |
| **Implement** | L2 regularization (weight decay) · Plot train/val loss vs lambda · K-Fold CV từ scratch |
| **Deliverable** | `day03_regularization.py` + bias-variance tradeoff plot |

---

#### Ngày 4 · Code: Decision Trees + Tree Ensembles (ML Spec Course 2 Week 4)

| | |
|---|---|
| **Resource** | [Kaggle Learn - Intro to ML](https://www.kaggle.com/learn/intro-to-machine-learning) (bài Decision Trees + Random Forest) |
| **Implement** | Dùng sklearn DecisionTreeClassifier + RandomForest · Tune max_depth · Feature importance bar plot |
| **Không cần** | Implement Decision Tree từ scratch (tốn thời gian, không trong scope) |
| **Deliverable** | `day04_trees.py` + feature importance visualization |

---

#### Ngày 5 · Code: Clustering + RL Intro (ML Spec Course 3)

| | |
|---|---|
| **Clustering** | K-Means từ scratch (numpy): khởi tạo centroids → assign → update → lặp |
| **RL Intro** | Chạy `gymnasium.make("FrozenLake-v1")` · Random policy · In reward trung bình |
| **Resource** | [D2L Ch.17.1 MDP](https://d2l.ai/chapter_reinforcement-learning/mdp.html) (đọc nhanh 20 phút) |
| **Deliverable** | `day05_kmeans_rl_intro.py` |

---

### 🔴 Deep Learning Specialization Coding (Ngày 6–15)

---

#### Ngày 6 · PyTorch Basics + DLS Course 1 Week 2
*Logistic Regression as Neural Network — lần đầu code với PyTorch*

| | |
|---|---|
| **Resource** | [D2L Ch.2 Preliminaries](https://d2l.ai/chapter_preliminaries/index.html) |
| **Implement** | Tensor ops · Autograd · Viết lại Logistic Regression ngày 2 bằng `nn.Module` |
| **Mục tiêu** | Cùng kết quả với day02, nhưng dùng PyTorch hoàn toàn |
| **Deliverable** | `day06_pytorch_logistic.py` |

---

#### Ngày 7 · DLS Course 1 Week 3–4: Shallow → Deep Neural Network

| | |
|---|---|
| **Resource** | [D2L Ch.4 Linear Classification](https://d2l.ai/chapter_linear-classification/index.html) + [Ch.5 MLP](https://d2l.ai/chapter_multilayer-perceptrons/index.html) |
| **Implement** | MLP với `nn.Sequential` · Train/Val split · `DataLoader` · Plot learning curves |
| **Dataset** | Fashion-MNIST (torchvision) |
| **Deliverable** | `day07_mlp_fashionmnist.py` + accuracy curve |

---

#### Ngày 8 · DLS Course 2 Week 1: Regularization trong PyTorch

| | |
|---|---|
| **Resource** | [D2L Ch.5.6 Dropout](https://d2l.ai/chapter_multilayer-perceptrons/dropout.html) |
| **Implement** | Dropout · L2 weight decay trong optimizer · Train/Val curves showing overfitting → regularized |
| **Deliverable** | `day08_regularization_dl.py` + before/after overfitting plot |

---

#### Ngày 9 · DLS Course 2 Week 2: Optimization Algorithms

| | |
|---|---|
| **Resource** | [D2L Ch.12](https://d2l.ai/chapter_optimization/index.html) |
| **Implement** | SGD · Momentum · Adam — implement từ scratch bằng numpy, rồi dùng `torch.optim` |
| **Visualize** | Convergence speed của 3 optimizer trên cùng 1 loss landscape |
| **Deliverable** | `day09_optimizers.py` + convergence comparison plot |

> [!IMPORTANT]
> Adam là optimizer mặc định trong hầu hết mọi thứ ở Level 2 — hiểu kỹ tại sao.

---

#### Ngày 10 · DLS Course 2 Week 3: Batch Normalization + Hyperparameter Tuning

| | |
|---|---|
| **Resource** | [D2L Ch.8.5 BatchNorm](https://d2l.ai/chapter_convolutional-modern/batch-norm.html) |
| **Implement** | Thêm BatchNorm vào MLP ngày 7 · So sánh training với/không có BatchNorm |
| **Tune** | Learning rate · Batch size · Số layers — dùng vòng lặp đơn giản, không cần AutoML |
| **Deliverable** | `day10_batchnorm_tuning.py` |

---

#### Ngày 11 · DLS Course 4 Week 1: CNN từ scratch → LeNet

| | |
|---|---|
| **Resource** | [D2L Ch.7.6 LeNet](https://d2l.ai/chapter_convolutional-neural-networks/lenet.html) |
| **Implement** | `nn.Conv2d` + `nn.MaxPool2d` · Build LeNet · Train trên MNIST |
| **Hiểu** | Feature maps: visualize output của mỗi conv layer |
| **Deliverable** | `day11_lenet.py` + feature map visualization |

---

#### Ngày 12 · DLS Course 5 Week 1: RNN + LSTM

| | |
|---|---|
| **Resource** | [D2L Ch.9.4-9.6](https://d2l.ai/chapter_recurrent-neural-networks/rnn.html) |
| **Implement** | `nn.RNN` → `nn.LSTM` · Character-level language model đơn giản |
| **Bỏ qua** | Implement BPTT từ scratch (DLS SKIP note: đã học ở DLS C5W1) |
| **Deliverable** | `day12_lstm_charmodel.py` |

---

#### Ngày 13 · DLS Course 5 Week 2: Word Embeddings

| | |
|---|---|
| **Resource** | [D2L Ch.15.1-15.4](https://d2l.ai/chapter_natural-language-processing-pretraining/word2vec.html) |
| **Implement** | Load pretrained GloVe · Cosine similarity · Word analogy: king - man + woman = ? |
| **Visualize** | t-SNE của word embeddings |
| **Deliverable** | `day13_word_embeddings.py` + t-SNE plot |

---

#### Ngày 14 · DLS Course 5 Week 4 + GenAI Week 1: Transformer

| | |
|---|---|
| **Resource** | [D2L Ch.11.7 Transformer](https://d2l.ai/chapter_attention-mechanisms-and-transformers/transformer.html) |
| **Implement** | Multi-Head Attention với `nn.MultiheadAttention` · Positional Encoding · Transformer Encoder block |
| **Visualize** | Attention heatmap |
| **Deliverable** | `day14_transformer_encoder.py` + attention heatmap |

---

#### Ngày 15 · GenAI Week 2: Fine-tuning + PEFT (LoRA nhẹ)

| | |
|---|---|
| **Resource** | HuggingFace `transformers` + `peft` library |
| **Implement** | Load `bert-base-uncased` · Fine-tune cho text classification · So sánh full fine-tune vs freeze backbone |
| **Deliverable** | `day15_finetune_bert.py` |

---

## 📅 LEVEL 2 — NLP · RAG · CV Coding (Ngày 16–30)

---

### 🟢 NLP Track Coding (Ngày 16–21)

---

#### Ngày 16 · NLP Course 1 Week 2: Naïve Bayes Classifier

| | |
|---|---|
| **Implement** | Naïve Bayes từ scratch: đếm frequency → prior/likelihood → log-likelihood → predict |
| **Dataset** | Sentiment analysis (Twitter dataset từ NLTK) |
| **Note** | Week 1 (Logistic Reg) đã học ở Level 1 — SKIP |
| **Deliverable** | `day16_naive_bayes.py` |

---

#### Ngày 17 · NLP Course 1 Week 3–4: Vector Space + Word Translation

| | |
|---|---|
| **Implement** | TF-IDF từ scratch · Cosine similarity cho document search · PCA để visualize word vectors 2D |
| **Bonus** | Word translation bằng linear transformation (alignment matrix) |
| **Deliverable** | `day17_vector_space.py` + PCA word cluster plot |

---

#### Ngày 18 · NLP Course 2 Week 2–3: HMM + N-gram Language Model

| | |
|---|---|
| **HMM** | Viterbi algorithm cho POS tagging — implement forward + backward pass |
| **N-gram** | Bigram language model từ scratch · Perplexity evaluation |
| **Deliverable** | `day18_hmm_ngram.py` |

---

#### Ngày 19 · NLP Course 2 Week 4: Word2Vec (CBOW)

| | |
|---|---|
| **Implement** | CBOW model: sliding window → one-hot → embedding → softmax |
| **Train** | Trên small corpus |
| **Compare** | Embedding của CBOW vs GloVe pretrained (ngày 13) |
| **Deliverable** | `day19_word2vec_cbow.py` |

---

#### Ngày 20 · NLP Course 3 Week 2: LSTM + NER

| | |
|---|---|
| **Resource** | HuggingFace `datasets` — CoNLL-2003 NER dataset |
| **Implement** | Bi-LSTM cho Named Entity Recognition · Train/eval với F1 per entity class |
| **Deliverable** | `day20_bilstm_ner.py` |

---

#### Ngày 21 · NLP Course 4 Week 1–2: Seq2Seq + Transformer NMT

| | |
|---|---|
| **Implement** | Encoder-Decoder với Attention · BLEU score evaluation |
| **Dùng** | HuggingFace `MarianMTModel` để baseline · So sánh với model tự train |
| **Deliverable** | `day21_seq2seq_nmt.py` |

---

### 🟡 RAG Track Coding (Ngày 22–25)

---

#### Ngày 22 · RAG Module 1–2: TF-IDF + BM25 Search

| | |
|---|---|
| **Implement** | TF-IDF retrieval từ scratch · BM25 retrieval (`rank_bm25` library) |
| **Task** | Build simple document search engine: query → top-K relevant docs |
| **Deliverable** | `day22_tfidf_bm25.py` |

---

#### Ngày 23 · RAG Module 2–3: Semantic Search + Vector DB

| | |
|---|---|
| **Implement** | Sentence embeddings với `sentence-transformers` · FAISS index · Cosine/ANN search |
| **Compare** | BM25 vs Semantic Search — khi nào cái nào tốt hơn? |
| **Deliverable** | `day23_semantic_search.py` |

---

#### Ngày 24 · RAG Module 3–4: Chunking + Reranking + Prompt Engineering

| | |
|---|---|
| **Implement** | Text chunking strategies (fixed, sentence, semantic) · Cross-encoder reranking |
| **Integrate** | Retriever + LLM (dùng OpenAI API hoặc local Ollama) · Build mini RAG pipeline |
| **Deliverable** | `day24_rag_pipeline.py` |

---

#### Ngày 25 · RAG Module 5: Evaluation + Mini-project RAG

| | |
|---|---|
| **Implement** | RAGAS-style evaluation: faithfulness + answer relevancy |
| **Mini-project** | RAG Q&A trên một domain nhỏ (medical papers / course notes) |
| **Deliverable** | `week4_rag_qa/` — README + demo · **Push GitHub** |

---

### 🟣 Computer Vision Track Coding (Ngày 26–30)

---

#### Ngày 26 · CV Unit 1–2: CNN Architectures + Transfer Learning

| | |
|---|---|
| **Implement** | Load VGG/ResNet/MobileNet từ `torchvision.models` · So sánh params, FLOPs, accuracy |
| **Fine-tune** | ResNet-18 trên small image dataset (CIFAR-10 hoặc medical) |
| **Deliverable** | `day26_cnn_transfer.py` + accuracy comparison table |

---

#### Ngày 27 · CV Unit 3: Vision Transformer (ViT)

| | |
|---|---|
| **Implement** | Patch embedding · ViT Encoder với `nn.MultiheadAttention` |
| **Dùng** | HuggingFace `ViTForImageClassification` · Fine-tune trên CIFAR-10 |
| **Compare** | CNN (ResNet) vs ViT — accuracy + speed |
| **Deliverable** | `day27_vit.py` |

---

#### Ngày 28 · CV Unit 4: Multimodal Models — CLIP

| | |
|---|---|
| **Dùng** | `openai/clip-vit-base-patch32` từ HuggingFace |
| **Task 1** | Zero-shot image classification: không train, chỉ dùng text prompt |
| **Task 2** | Image-text similarity search |
| **Task 3** | Visualize attention maps của CLIP |
| **Deliverable** | `day28_clip_multimodal.py` + zero-shot demo |

---

#### Ngày 29 · CV Unit 5: Generative Models — VAE + Diffusion intro

| | |
|---|---|
| **VAE** | Implement VAE nhỏ với PyTorch · Train trên MNIST · Visualize latent space interpolation |
| **Diffusion** | Dùng `diffusers` library: load Stable Diffusion nhỏ · Text-to-image inference |
| **Không cần** | Implement diffusion từ scratch |
| **Deliverable** | `day29_vae_diffusion.py` + generated samples |

---

#### Ngày 30 · Final: Tổng kết + GitHub cleanup

| | |
|---|---|
| **Việc làm** | Viết README tổng cho repo `lab-level1-2` · Document 5 deliverable quan trọng nhất |
| **Review** | Chạy lại 3 code quan trọng: Transformer (ngày 14) · CLIP (ngày 28) · RAG pipeline (ngày 24) |
| **Chuẩn bị** | Note những concept còn mơ hồ → hỏi mentor khi vào lab |
| **Deliverable** | Repo GitHub sạch với README tổng + `requirements.txt` |

---

## 📊 Timeline tổng hợp

```
Ngày  1- 5  │ 🔵 ML Spec    │ Linear Reg → Log Reg → Regularization → Trees → Clustering + RL intro
Ngày  6-10  │ 🔴 DL Spec ①  │ PyTorch basics → MLP → Regularization DL → Optimization → BatchNorm
Ngày 11-15  │ 🔴 DL Spec ②  │ CNN/LeNet → RNN/LSTM → Word Embeddings → Transformer → Fine-tune BERT
Ngày 16-21  │ 🟢 NLP Track  │ Naïve Bayes → Vector Space → HMM → Word2Vec → NER → Seq2Seq
Ngày 22-25  │ 🟡 RAG Track  │ TF-IDF/BM25 → Semantic Search → RAG pipeline → Evaluation
Ngày 26-30  │ 🟣 CV Track   │ CNN Architectures → ViT → CLIP Multimodal → VAE → Final cleanup
```

---

## 🔗 Resource mapping

| Ngày | Resource chính |
|---|---|
| 1–3 | [Google MLCC Exercises](https://developers.google.com/machine-learning/crash-course/exercises) |
| 4–5 | [Kaggle Learn](https://www.kaggle.com/learn) |
| 6–14 | [D2L.ai](https://d2l.ai) |
| 15, 20, 21, 27, 28 | [HuggingFace](https://huggingface.co) |
| 16–21 | NLP Specialization (deeplearning.ai) |
| 22–25 | RAG course (deeplearning.ai) |
| 26–29 | CV course + torchvision |
| Tra cứu toán | [machinelearningcoban.com](https://machinelearningcoban.com) + [deepai-book](https://phamdinhkhanh.github.io/deepai-book/intro.html) |

---

## 💡 Nguyên tắc 30 ngày

> [!CAUTION]
> **Không đọc nhiều hơn code.** Mỗi ngày tối đa 30 phút đọc — còn lại là gõ code.

1. **Push GitHub mỗi ngày** — kể cả code chưa xong
2. **Không copy-paste** — gõ lại từng dòng kể cả khi nhìn tutorial
3. **Chạy được = done** — đừng perfectionism, deliverable chỉ cần chạy và in kết quả
4. Khi bị stuck > 20 phút — hỏi ngay, đừng tự block

