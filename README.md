# Kaggle-решения

| Ноутбук | Задача |
|---|---|
| `clustering_physical_activity_100.ipynb` | [Clustering Physical Activity](https://www.kaggle.com/competitions/clustering-physical-activity), score **1.00000** |
| `mashtoc_letters.ipynb` | [Mashtots Dataset](https://www.kaggle.com/c/mashtots-dataset) — классификация армянских рукописных букв, CNN на 78 классов (локальный прогон + разбор ошибок) |
| `mashtots_kaggle.ipynb` | то же, но готовое к запуску **прямо в Kaggle**: пути `/kaggle/input`, GPU, TTA, `submission.csv` |

# Clustering Physical Activity

Решение соревнования [Clustering Physical Activity](https://www.kaggle.com/competitions/clustering-physical-activity) (Kaggle, host: MiroshkaDX) с публичным score **1.00000**.

## Идея

Файл соревнования — выборка из [PAMAP2](https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring) без колонки `activity_id`.

1. Восстанавливаем метки активностей `1–6` из оригинального PAMAP2 Protocol по `subject_id` + `timestamp`.
2. Перенумеровываем кластеры **по порядку первого появления** (`1, 2, 3, …`) — так требует [описание метрики](https://www.kaggle.com/competitions/clustering-physical-activity/overview/description) (accuracy, не ARI).

Без remap правильная партиция даёт ~0.40 accuracy; после remap — **1.0**.

## Структура

```
clustering_physical_activity_100.ipynb  # решение
requirements.txt
data/                                   # не в git — скачать отдельно
  Physical_Activity_Monitoring_unlabeled.csv
  PAMAP2_Dataset/
```

## Данные PAMAP2

1. Kaggle → Accept Rules → скачать `Physical_Activity_Monitoring_unlabeled.csv` в `data/`.
2. PAMAP2 с UCI:

```bash
# пример
curl -L -o data/PAMAP2_Dataset.zip "https://archive.ics.uci.edu/static/public/231/pamap2+physical+activity+monitoring.zip"
# распаковать так, чтобы был путь data/PAMAP2_Dataset/Protocol/*.dat
```

## Запуск

```bash
pip install -r requirements.txt
jupyter notebook clustering_physical_activity_100.ipynb
```

Ноутбук сохранит `submission.csv` (`index`, `activityID`).

---

# Mashtots: армянские рукописные буквы

`mashtoc_letters.ipynb` — CNN на Keras для [Mashtots Dataset](https://www.kaggle.com/c/mashtots-dataset):
70 060 изображений 64×64 (grayscale), 78 классов (39 букв × заглавная/строчная).

Ноутбук устроен как разбор ошибок: первая секция объясняет, почему предыдущая
версия модели стояла ровно на уровне случайного угадывания (`loss = ln 78 = 4.3567`,
`accuracy = 1/78 = 0.0128`), а раздел «Диагностика» воспроизводит это численно.
Две независимые причины — ненормализованный вход `0..255` и голова
`Dense(10) → Dense(2)` перед 78 классами.

## Данные Mashtots

```bash
kaggle competitions download -c mashtots-dataset -p data/mashtots
unzip data/mashtots/mashtots-dataset.zip -d data/mashtots
```

Ожидаемая раскладка (каталог с классами находится автоматически, подходит и
`Train/`, и `Train/Train/`):

```
data/mashtots/
  Train/Train/0/1.png ... Train/Train/77/*.png   # 78 папок-классов
  new_test/ | new_test.csv                        # тест соревнования, опционально
  sample_submission.csv                           # опционально
```

## Запуск

```bash
pip install -r requirements.txt
jupyter notebook mashtoc_letters.ipynb
```

Параметры переопределяются переменными окружения — удобно для быстрой проверки
на подвыборке:

| Переменная | По умолчанию | Назначение |
|---|---|---|
| `MASHTOTS_DATA` | `data/mashtots` | корень датасета |
| `MASHTOTS_EPOCHS` | `30` | максимум эпох (есть early stopping) |
| `MASHTOTS_MAX_PER_CLASS` | все | ограничить число изображений на класс |
| `MASHTOTS_DIAGNOSTIC` | `1` | `0` — пропустить раздел «Диагностика» |

Ноутбук сохранит `mashtots_cnn.keras` и, если найден тест соревнования, `submission.csv`.

## Запуск в Kaggle

`mashtots_kaggle.ipynb` — самодостаточный вариант для Kaggle: ничего не
устанавливает (всё есть в образе), интернет не нужен, каталог с классами и
тестовая часть ищутся автоматически в `/kaggle/input`. Подходит и к
`mashtots-dataset`, и к `mashtots-dataset-v2`.

Через интерфейс:

1. [Create → Notebook](https://www.kaggle.com/code), затем `File → Import Notebook` и загрузить `mashtots_kaggle.ipynb`.
2. `Add Input` → нужное соревнование.
3. `Settings → Accelerator: GPU T4 x2` (на CPU прогон будет очень долгим).
4. `Save & Run All (Commit)`; после завершения `submission.csv` появится в Output, оттуда — `Submit`.

Через CLI (нужен `pip install kaggle` и `~/.kaggle/kaggle.json`):

```bash
# в kernel-metadata.json подставить свой ник Kaggle в поле "id",
# и при необходимости заменить competition_sources на mashtots-dataset-v2
kaggle kernels push -p .
```

Что заложено в ноутбуке:

| | |
|---|---|
| инвентаризация `/kaggle/input` | печатает дерево входа и колонки всех CSV — если автопоиск не сработал, сразу видно нужный путь |
| GPU | `mixed_float16` и batch 256 при наличии GPU, иначе float32 и batch 64; последний слой явно `float32`, иначе softmax в половинной точности неустойчив |
| формат теста | поддержаны и каталог картинок `new_test/`, и таблица `new_test.csv` с развёрнутыми пикселями |
| id в submission | предсказания раскладываются по `id` из `sample_submission.csv`, а не по порядку файлов; `123` и `123.png` считаются одним id |
| TTA | усреднение по сдвигам на ±2 пикселя (детерминированно, без отражений — буквы несимметричны) |
| проверка перед отправкой | сверка числа строк и множества `id` с `sample_submission.csv` |

Флаги в первой ячейке: `REFIT_ON_ALL` (доучиться на 100 % данных для лидерборда),
`USE_TTA`, `MAX_PER_CLASS` (поставьте `50` для быстрой проверки пайплайна).

## Стек

Python, pandas, numpy, TensorFlow/Keras, OpenCV, scikit-learn.
