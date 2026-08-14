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

## Данные

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

## Стек

Python, pandas, numpy.
