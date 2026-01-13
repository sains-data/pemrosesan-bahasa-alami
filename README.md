# Mata Kuliah: Pemrosesan Bahasa Alami (Natural Language Processing)

Repository ini berisi materi, notebook praktikum, kode sumber, dan referensi untuk mata kuliah Pemrosesan Bahasa Alami (Natural Language Processing / NLP). Mata kuliah ini membahas konsep dasar hingga lanjutan NLP, mulai dari pendekatan klasik berbasis statistik dan machine learning hingga model deep learning dan large language models (LLM) modern.

## Capaian Pembelajaran
Setelah mengikuti mata kuliah ini, mahasiswa diharapkan:
1. Mampu menjelaskan ruang lingkup, tantangan, dan aplikasi pemrosesan bahasa alami dalam pengolahan data berbasis teks.
2. Mampu mengidentifikasi dan mengekstraksi fitur penting dari teks menggunakan metode representasi data.
3. Mampu menerapkan teknik pra-pemrosesan teks serta membandingkan representasi vektor kata dan language models.
4. Mampu mengimplementasikan algoritma POS tagging dan N-gram untuk pemodelan struktur bahasa alami.
5. Mampu mengklasifikasikan teks dan menerapkan teknik text mining untuk menemukan informasi tersembunyi dalam data teks.
6. Mampu mengevaluasi dan menerapkan algoritma machine learning yang relevan dalam pemrosesan teks.
7. Mampu menjelaskan dan mengimplementasikan dependency parsing untuk memahami struktur sintaksis kalimat.
8. Mampu mengembangkan dan menerapkan model Recurrent Neural Network dan arsitektur turunannya untuk tugas-tugas pemrosesan bahasa alami.
9. Mampu menerapkan metode post-training (Reinforcement Learning from Human Feedback, Supervised Fine-Tuning, Direct Preference Optimization) menggunakan model pretrained dalam penyempurnaan model pemrosesan bahasa alami.		
10. Mampu mengevaluasi dan mengadaptasi metode efisien seperti prompting dan Parameter Efficient Fine Tuning untuk mempercepat adaptasi model pemrosesan bahasa alami.		
11. Mampu melakukan evaluasi dan benchmarking model pemrosesan bahasa alami berdasarkan metrik kinerja standar industri.
Mampu mengembangkan sistem Question Answering berbasis Natural language processing dengan integrasi pengetahuan domain.

## Topik Perkuliahan

1. **Pengantar Pemrosesan Bahasa Alami (NLP)**

-  Definisi dan ruang lingkup NLP dalam kecerdasan buatan
-  Tantangan utama NLP (ambiguity, sparsity, dll.)
-  Aplikasi NLP di dunia nyata
-  Arsitektur dasar sistem NLP: pipeline klasik vs end-to-end learning

2. **Teknik Ekstraksi Fitur dalam Teks Bahasa Alami**

-  Representasi teks: Bag-of-Words (BoW) dan TF-IDF
-  One-hot encoding dan keterbatasannya
-  Feature selection dalam klasifikasi teks
-  Implementasi ekstraksi fitur pada dataset teks sederhana

3. **Pra-pemrosesan Teks, Word Embeddings, dan Language Models**

-  Tokenisasi, stemming, lemmatization, stopword removal
-  Word embeddings: Word2Vec, GloVe, fastText
-  Praktik menggunakan NLTK dan spaCy
-  Language model: n-gram vs pretrained models

4. **POS Tagging dan Pemodelan Tag N-Gram**

-  Konsep dan tujuan POS tagging
-  N-gram model dan Hidden Markov Model (HMM)
-  Dataset POS tagging (misalnya Penn Treebank)
-  Evaluasi dan akurasi POS tagger

5. **Sistem Information Retrieval (IR)**

-  Konsep dan arsitektur sistem IR
-  Relevansi dan perankingan dokumen
-  Evaluasi IR: precision, recall, F1-score, MAP
-  Aplikasi IR: search engine, Q&A system, recommendation

6. **Teknik Labeling dan Aplikasi Text Mining**

-  Text annotation: manual vs otomatis
-  Named Entity Recognition (NER)
-  Text mining: klasifikasi, klastering, analisis sentimen
-  Tools: spaCy, NLTK, Prodigy

7. **Algoritma NLP dalam Machine Learning**

-  Representasi fitur untuk supervised learning
-  Algoritma klasifikasi: Naive Bayes, SVM, Decision Tree
-  Unsupervised NLP: clustering dan topic modeling (LDA)
-  Integrasi NLP–ML dalam pipeline prediksi teks

8. **Ujian Tengah Semester (UTS)**

9. **Dependency Parsing dan Analisis Sintaksis**

-  Konsep dependency vs constituency parsing
-  Dependency tree dan relasi sintaksis
-  Algoritma parsing: transition-based dan graph-based
-  Implementasi parser dengan spaCy atau Stanza

10. **Deep Learning untuk NLP**

-  Peran deep learning dalam NLP
-  Arsitektur RNN, LSTM, dan GRU
-  Model sequence-to-sequence (seq2seq)
-  Perbandingan pendekatan tradisional vs deep learning

11. **Transformer dan Model Pretrained**

-  Self-attention dan positional encoding
-  Encoder–decoder Transformer
-  Model pretrained: BERT, GPT, RoBERTa
-  Fine-tuning untuk tugas NLP spesifik

12. **Post-Training Model Bahasa Besar**

-  Motivasi post-training
-  Reinforcement Learning from Human Feedback (RLHF)
-  Supervised Fine-Tuning (SFT)
-  Direct Preference Optimization (DPO)

13. **Adaptasi Efisien Model Bahasa**

-  Prompt engineering
-  Zero-shot, one-shot, few-shot prompting
-  Parameter-Efficient Fine-Tuning (PEFT): LoRA, Adapter, Prefix Tuning
-  Perbandingan fine-tuning penuh vs PEFT

14. **Evaluasi dan Benchmarking Model NLP**

-  Metrik evaluasi: accuracy, F1, BLEU, ROUGE, perplexity
-  Evaluasi tugas klasifikasi, ekstraksi, dan generasi
-  Benchmark: GLUE, SuperGLUE, SQuAD, HELM
-  Tantangan evaluasi model skala besar

15. **Question Answering dan Representasi Pengetahuan**

-  Tipe sistem QA: open-domain vs closed-domain
-  Pipeline QA: retrieval dan reader/generator
-  Knowledge graph dan knowledge embedding
-  Integrasi QA dengan LLM dan sumber eksternal

16. **Ujian Akhir Semester (UAS)**

## Teknologi dan Library yang Digunakan

1. Python 3.x
2. NLTK
3. spaCy
4. scikit-learn
5. PyTorch
6. HuggingFace Transformers
7. Jupyter Notebook

## Struktur Repository
```
.
├── 01-pengantar-nlp/
├── 02-ekstraksi-fitur/
├── 03-preprocessing-embeddings/
├── 04-pos-tagging/
├── 05-information-retrieval/
├── 06-text-mining/
├── 07-ml-for-nlp/
├── 09-dependency-parsing/
├── 10-deep-learning-nlp/
├── 11-transformer/
├── 12-post-training/
├── 13-peft-prompting/
├── 14-evaluasi-benchmark/
├── 15-question-answering/
├── referensi/
└── README.md
```

## Referensi Utama

1. Jurafsky, D., & Martin, J. H. Speech and Language Processing: An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition.
2. Lee, R. S. (2023). Natural Language Processing. In Natural Language Processing: A Textbook with Python Implementation (pp. 3-17). Singapore: Springer Nature Singapore.
3. Haralambous, Y. A Course in Natural.
4. Dahl, D. A. (2023). Natural Language Understanding with Python: Combine natural language technology, deep learning, and large language models to create human-like language comprehension in computer systems. Packt Publishing Ltd.
5. Esposito, F. (2024). Programming Large Language Models with Azure Open AI: Conversational Programming and Prompt Engineering with LLMs. Microsoft Press.

## Catatan

Repository ini bersifat edukatif dan dikembangkan untuk mendukung proses pembelajaran dan praktikum mata kuliah Pemrosesan Bahasa Alami.
Silakan gunakan sesuai kebutuhan akademik dan kontribusi melalui pull request sangat kami hargai.