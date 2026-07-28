# Отчет по летней практике: Обзор задач Kaggle (Machine Learning, CV, NLP)

Данная документация содержит описание задач, форматов данных, целевых метрик и базовых подходов, изученных и реализованных в рамках прохождения практики.

---

## 📌 Структура и направления практики

В ходе практики проработаны 11 практических соревнований с платформы Kaggle по трем ключевым направлениям:
1. **Tabular Machine Learning** — Работа с табличными данными, Feature Engineering, градиентный бустинг.
2. **Computer Vision (CV)** — Обработка изображений, сверточные нейронные сети (CNN), трансформеры (ViT) и детектирование объектов (YOLO/Faster R-CNN).
3. **Natural Language Processing (NLP)** — Анализ текста, TF-IDF, эмбеддинги и Fine-tuning трансформеров (BERT/RoBERTa).

---

## 🛠 Подробное описание задач

### 1. Табулярное машинное обучение (Tabular ML)

#### 1.1 Titanic - Machine Learning from Disaster
* **Ссылка:** [Kaggle Titanic](https://www.kaggle.com/c/titanic)
* **Цель:** Предсказать выживаемость пассажиров Титаника на основе демографических данных и информации о билете.
* **Тип задачи:** Бинарная классификация (Binary Classification).
* **Входные данные:** Таблица с характеристиками пассажиров (`Age`, `Sex`, `Pclass`, `Fare`, `SibSp`, `Parch`, `Cabin`, `Embarked`).
* **Формат ответа (Target):** Таблица `PassengerId, Survived` (1 — выжил, 0 — погиб).
* **Метрика качества:** Accuracy.
* **Основные методы:**
  * Анализ и импутация пропущенных значений (`Age`, `Embarked`, `Fare`).
  * Feature Engineering: вычленение титулов из имени (`Mr`, `Mrs`, `Miss`, `Master`), создание признака `FamilySize` и индикатора `IsAlone`.
  * Модели: Random Forest, LightGBM, Logistic Regression.

#### 1.2 House Prices - Advanced Regression Techniques
* **Ссылка:** [Kaggle House Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
* **Цель:** Спрогнозировать итоговую стоимость жилого дома.
* **Тип задачи:** Регрессия (Regression).
* **Входные данные:** Таблица из 79 признаков, описывающих характеристики объекта недвижимости (площадь, этажность, материал крыши, год постройки, состояние фундамента и др.).
* **Формат ответа (Target):** Таблица `Id, SalePrice` (стоимость в USD).
* **Метрика качества:** RMSLE (Root Mean Squared Logarithmic Error).
* **Основные методы:**
  * Логарифмирование целевой переменной `log1p(SalePrice)` для устранения скошенности распределения.
  * Кодирование категориальных признаков (One-Hot Encoding, Target Encoding).
  * Борьба с мультиколлинеарностью и ансамблирование регрессоров (Ridge, Lasso, XGBoost, LightGBM, CatBoost).

#### 1.3 Home Credit Default Risk
* **Ссылка:** [Kaggle Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk)
* **Цель:** Оценить вероятность того, что заемщик не сможет выплатить кредит (дефолт).
* **Тип задачи:** Бинарная классификация с сильным дисбалансом классов.
* **Входные данные:** Реляционная база данных историй платежей, прошлых заявок и данных кредитного бюро (`bureau`, `previous_application`, `POS_CASH_balance`, `installments_payments`, `credit_card_balance`).
* **Формат ответа (Target):** Таблица `SK_ID_CURR, TARGET` (вероятность от 0.0 до 1.0).
* **Метрика качества:** ROC-AUC.
* **Основные методы:**
  * Расчет агрегированных статистических признаков по связным реляционным таблицам (Mean, Max, Min, Std, Sum).
  * Обработка дисбаланса классов (`scale_pos_weight`, SMOTE) и отбор наиболее значимых фичей.
  * Обучение ансамблей градиентного бустинга (LightGBM / CatBoost / XGBoost).

#### 1.4 Santander Customer Transaction Prediction
* **Ссылка:** [Kaggle Santander Customer Transaction Prediction](https://www.kaggle.com/c/santander-customer-transaction-prediction)
* **Цель:** Определить, совершит ли клиент определенную транзакцию в будущем независимо от ее суммы.
* **Тип задачи:** Бинарная классификация.
* **Входные данные:** Анонимизированный табличный датасет из 200 числовых признаков (`var_0` – `var_199`) без контекста и названий.
* **Формат ответа (Target):** Таблица `ID_code, target` (вероятность 0 или 1).
* **Метрика качества:** ROC-AUC.
* **Основные методы:**
  * Поиск и очистка синтетических (сгенерированных) данных в тестовой выборке.
  * Выделение частотных признаков (Count Encoding уникальных и повторяющихся значений).
  * Попризнаковое обучение независимых моделей и применение LightGBM / CatBoost.

---

### 2. Компьютерное зрение (Computer Vision)

#### 2.1 Digit Recognizer (MNIST)
* **Ссылка:** [Kaggle Digit Recognizer](https://www.kaggle.com/c/digit-recognizer)
* **Цель:** Распознать рукописную цифру от 0 до 9 по ее пиксельному изображению.
* **Тип задачи:** Многоклассовая классификация (Multiclass Classification, 10 классов).
* **Входные данные:** Градационные изображения размером 28x28 пикселей, представленные в виде 784 столбцов яркости.
* **Формат ответа (Target):** Таблица `ImageId, Label` (целое число от 0 до 9).
* **Метрика качества:** Accuracy.
* **Основные методы:**
  * Решейпинг (Reshape) плоского вектора в 2D-тензор `(28, 28, 1)` и нормализация значений пикселей в диапазон `[0, 1]`.
  * Аугментация данных (небольшие повороты, случайные сдвиги и масштабирование).
  * Построение глубокой сверточной нейронной сети (CNN) с BatchNorm, Dropout и MaxPool.

#### 2.2 Cassava Leaf Disease Classification
* **Ссылка:** [Kaggle Cassava Leaf Disease Classification](https://www.kaggle.com/c/cassava-leaf-disease-classification)
* **Цель:** Классифицировать тип заболевания на листьях кассавы по фотографиям.
* **Тип задачи:** Многоклассовая классификация (5 классов: 4 болезни + 1 здоровое растение).
* **Входные данные:** Реальные фотографии листьев с полей, сделанные в неидеальных условиях освещения и ракурса.
* **Формат ответа (Target):** Таблица `image_id, label` (индекс класса от 0 до 4).
* **Метрика качества:** Accuracy.
* **Основные методы:**
  * Использование современных CV-архитектур: EfficientNet, ResNeXt, Vision Transformers (ViT).
  * Продвинутые аугментации: CutMix, Mixup, RandomResizedCrop (Albumentations).
  * Борьба с шумными метками в данных с помощью Label Smoothing Cross-Entropy.

#### 2.3 Global Wheat Detection
* **Ссылка:** [Kaggle Global Wheat Detection](https://www.kaggle.com/c/global-wheat-detection)
* **Цель:** Локализовать и распознать пшеничные колоски на фотографиях пшеничных полей.
* **Тип задачи:** Детектирование объектов (Object Detection).
* **Входные данные:** Изображения высокого разрешения пшеничных полей с разнородным фоном, освещением и степенью перекрытия объектов.
* **Формат ответа (Target):** Таблица `image_id, PredictionString` (ограничивающие рамки в формате `confidence xmin ymin width height`).
* **Метрика качества:** mean Average Precision (mAP) при различных порогах IoU (Intersection over Union от 0.5 до 0.75).
* **Основные методы:**
  * Форматирование аннотаций ограничивающих рамок (Bounding Boxes) и аугментация изображений (Random Sized Crop, Brightness Contrast, Mosaic Augmentation via Albumentations).
  * Применение моделей детектирования: Faster R-CNN, EfficientDet, YOLOv5.
  * Постобработка предсказаний с помощью WBF (Weighted Boxes Fusion) вместо классического NMS (Non-Maximum Suppression) для объединения рамок от разных моделей.

---

### 3. Обработка естественного языка (Natural Language Processing)

#### 3.1 Bag of Words Meets Bags of Popcorn (word2vec-nlp-tutorial)
* **Ссылка:** [Kaggle Word2Vec Tutorial](https://www.kaggle.com/c/word2vec-nlp-tutorial)
* **Цель:** Классифицировать эмоциональную окраску (sentiment) отзывов к фильмам на IMDb.
* **Тип задачи:** Бинарная классификация текстов.
* **Входные данные:** Неструктурированные текстовые отзывы на английском языке.
* **Формат ответа (Target):** Таблица `id, sentiment` (1 — позитивный, 0 — негативный).
* **Метрика качества:** ROC-AUC.
* **Основные методы:**
  * Очистка текста: удаление HTML-тегов, пунктуации, стоп-слов, токенизация и лемматизация.
  * Сравнение методов векторизации: TF-IDF vs Word2Vec / FastText с использованием Logistic Regression и Naive Bayes.

#### 3.2 Natural Language Processing with Disaster Tweets
* **Ссылка:** [Kaggle NLP Disaster Tweets](https://www.kaggle.com/c/nlp-getting-started)
* **Цель:** Определить, сообщает ли твит о реальной чрезвычайной ситуации или использует слова-триггеры в метафорическом контексте.
* **Тип задачи:** Бинарная классификация коротких текстов.
* **Входные данные:** Тексты твитов (до 280 символов) с хэштегами, меншенами и опечатками + опциональные метаданные.
* **Формат ответа (Target):** Таблица `id, target` (1 — реальная катастрофа, 0 — обычный твит).
* **Метрика качества:** F1-Score.
* **Основные методы:**
  * Нормализация специфичного веб-текста (удаление URL, нормализация эмодзи и хэштегов).
  * Fine-tuning предобученных языковых моделей-трансформеров (BERT, RoBERTa) с использованием Hugging Face Transformers.

#### 3.3 Jigsaw Toxic Comment Classification Challenge
* **Ссылка:** [Kaggle Jigsaw Toxic Comment](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)
* **Цель:** Выявить и классифицировать различные типы токсичности в комментариях Wikipedia.
* **Тип задачи:** Многоэтикеточная классификация (Multi-label Classification: 6 независимых категорий токсичности).
* **Входные данные:** Тексты комментариев пользователей.
* **Формат ответа (Target):** Таблица `id` и 6 столбцов вероятностей: `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`.
* **Метрика качества:** Mean ROC-AUC по всем 6 категориям.
* **Основные методы:**
  * Предобработка анонимизированного и агрессивного текста.
  * Рекуррентные сети (Bi-GRU / Bi-LSTM с Attention) и трансформеры (DeBERTa / RoBERTa).

#### 3.4 Quora Question Pairs
* **Ссылка:** [Kaggle Quora Question Pairs](https://www.kaggle.com/c/quora-question-pairs)
* **Цель:** Определить, являются ли два вопроса с платформы Quora дубликатами по смыслу.
* **Тип задачи:** Бинарная классификация пар текстов (Semantic Textual Similarity).
* **Входные данные:** Пары вопросов (`question1`, `question2`).
* **Формат ответа (Target):** Таблица `test_id, is_duplicate` (1 — вопросы дублируют друг друга, 0 — нет).
* **Метрика качества:** Log Loss.
* **Основные методы:**
  * Извлечение эвристических и семантических признаков (расстояние Левенштейна, коэффициент Жаккара, сходство TF-IDF векторов).
  * Использование сиамских нейронных сетей (Siamese Neural Networks) и Cross-Encoder архитектур на базе BERT.

#### 3.5 Tweet Sentiment Extraction
* **Ссылка:** [Kaggle Tweet Sentiment Extraction](https://www.kaggle.com/c/tweet-sentiment-extraction)
* **Цель:** Извлечь подстроку из текста твита, которая является причиной его тональности.
* **Тип задачи:** Извлечение фрагментов текста (Span Extraction / Question Answering).
* **Входные данные:** Текст твита и заранее известная тональность (`positive`, `negative`, `neutral`).
* **Формат ответа (Target):** Таблица `textID, selected_text` (извлеченная ключевая подстрока).
* **Метрика качества:** Jaccard Similarity Coefficient.
* **Основные методы:**
  * Формулирование задачи как SQuAD Question Answering (модель принимала Sentiment как вопрос и Tweet как контекст).
  * Предсказание индексов `start_logits` и `end_logits` с использованием RoBERTa / DeBERTa.

---

## 📊 Сводный обход метрик и стека

| Категория | Задача | Направленность | Метрика | Основной стек |
|---|---|---|---|---|
| **Tabular ML** | Titanic | Binary Classification | Accuracy | Pandas, Scikit-Learn, LightGBM |
| **Tabular ML** | House Prices | Regression | RMSLE | Ridge, XGBoost, CatBoost |
| **Tabular ML** | Home Credit Default Risk | Binary Classification | ROC-AUC | LightGBM, Relational Feature Engineering |
| **Tabular ML** | Santander Customer Transaction | Binary Classification | ROC-AUC | LightGBM, Frequency Encoding |
| **CV** | Digit Recognizer | Multiclass (10) | Accuracy | PyTorch, CNN, Torchvision |
| **CV** | Cassava Leaf Disease | Multiclass (5) | Accuracy | PyTorch, EfficientNet, Albumentations |
| **CV** | Global Wheat Detection | Object Detection | mAP (IoU 0.5:0.75) | PyTorch, Faster R-CNN / YOLOv5, WBF |
| **NLP** | Word2Vec Tutorial | Binary Classification | ROC-AUC | NLTK, TF-IDF, Word2Vec, Scikit-Learn |
| **NLP** | Disaster Tweets | Binary Classification | F1-Score | Hugging Face, BERT, RoBERTa |
| **NLP** | Jigsaw Toxic | Multi-label (6) | Mean ROC-AUC | PyTorch, Bi-GRU + Attention, RoBERTa |
| **NLP** | Quora Question Pairs | Semantic Similarity | Log Loss | Feature Engineering, Siamese Networks |
| **NLP** | Tweet Sentiment | Span Extraction (QA) | Jaccard Similarity | Hugging Face, RoBERTa, DeBERTa |

---

## 📁 Рекомендуемая структура репозитория

```text
.
├── README.md
├── tabular/
│   ├── 01_titanic.ipynb
│   ├── 02_house_prices.ipynb
│   ├── 03_home_credit.ipynb
│   └── 04_santander.ipynb
├── cv/
│   ├── 01_digit_recognizer.ipynb
│   ├── 02_cassava_leaf_disease.ipynb
│   └── 03_global_wheat_detection.ipynb
└── nlp/
    ├── 01_word2vec_tutorial.ipynb
    ├── 02_disaster_tweets.ipynb
    ├── 03_jigsaw_toxic.ipynb
    ├── 04_quora_question_pairs.ipynb
    └── 05_tweet_sentiment_extraction.ipynb
```
