# Kaggle-решения

| Проект | Соревнование |
|---|---|
| [`clustering_physical_activity_100.ipynb`](clustering_physical_activity_100.ipynb) | [Clustering Physical Activity](https://www.kaggle.com/competitions/clustering-physical-activity), публичный score **1.00000** |
| [`mashtots/`](mashtots/) | [Mashtots Dataset](https://www.kaggle.com/competitions/mashtots-dataset) — CNN на 78 классов армянских рукописных букв |

---

# Clustering Physical Activity

Решение соревнования [Clustering Physical Activity](https://www.kaggle.com/competitions/clustering-physical-activity) (Kaggle, host: MiroshkaDX) с публичным score **1.00000**.

## Идея

Файл соревнования — выборка из [PAMAP2](https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring) без колонки `activity_id`.

1. Восстанавливаем метки активностей `1–6` из оригинального PAMAP2 Protocol по `subject_id` + `timestamp`.
2. Перенумеровываем кластеры **по порядку первого появления** (`1, 2, 3, …`) — так требует [описание метрики](https://www.kaggle.com/competitions/clustering-physical-activity/overview/description) (accuracy, не ARI).

Без remap правильная партиция даёт ~0.40 accuracy; после remap — **1.0**.

## Данные

1. Kaggle → Accept Rules → скачать `Physical_Activity_Monitoring_unlabeled.csv` в `data/`.
2. PAMAP2 с UCI:

```bash
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

# Mashtots — армянские рукописные буквы

CNN на 78 классов (39 букв армянского алфавита × заглавная и строчная),
изображения 64×64 в градациях серого, 70 060 штук в обучающей части.

Решение целиком лежит в папке [`mashtots/`](mashtots/) — со своим README,
зависимостями и ноутбуками для Kaggle и локального прогона.

## Стек

Python, pandas, numpy, TensorFlow/Keras, OpenCV, scikit-learn.
