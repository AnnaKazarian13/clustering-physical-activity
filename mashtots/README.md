# Mashtots — классификация армянских рукописных букв

CNN на 78 классов для соревнований
[Mashtots Dataset](https://www.kaggle.com/competitions/mashtots-dataset) и
[Mashtots Dataset v2](https://www.kaggle.com/competitions/mashtots-dataset-v2):
39 букв армянского алфавита в двух начертаниях, изображения 64×64 в градациях
серого, 70 060 штук в обучающей части.

## Файлы

| Файл | Назначение |
|---|---|
| `mashtots_kaggle.ipynb` | запуск в Kaggle: пути `/kaggle/input`, GPU, TTA, `submission.csv` |
| `mashtots_local.ipynb` | локальный прогон: обучение, разбор ошибок, предсказание для одного файла |
| `kernel-metadata.json` | конфигурация для `kaggle kernels push` |
| `requirements.txt` | зависимости для локального запуска (в Kaggle не нужны) |

## Модель

`Rescaling` → аугментация → три блока `Conv-BN-ReLU ×2 → MaxPool → Dropout`
(32, 64, 128 фильтров) → `Flatten → Dense(256) → BN → Dropout → Dense(78)`,
около 2.5 М параметров. Adam `1e-3`, `sparse_categorical_crossentropy`,
`EarlyStopping` + `ReduceLROnPlateau`, стратифицированное разбиение 80 / 10 / 10.

Решения, которые стоит знать перед правками:

* работаем в родных 64×64: ресайз вверх не добавляет информации, но умножает
  время и память — 200×200 это 2.6 ГиБ вместо 274 МиБ на датасет;
* изображения хранятся в `uint8`, Keras всё равно приводит батч к float сам;
* метка класса берётся как `int(имя папки)` — при сортировке по строкам папка
  `10` встала бы сразу после `1`;
* нормализация и аугментация сделаны слоями внутри модели, поэтому
  препроцессинг обучения и инференса невозможно рассинхронизировать;
* аугментация без отражений: буквы зеркально несимметричны. Пустота после
  сдвига и поворота заливается нулями — фон датасета чёрный, и дефолтный
  `fill_mode="reflect"` затащил бы в кадр куски штриха;
* `BatchNormalization(momentum=0.9)`: при дефолтных `0.99` скользящие
  статистики сходятся только к ~2000 шагам;
* предсказания раскладываются по `id` из `sample_submission.csv`, а не по
  порядку файлов — каталог читается лексикографически (`1, 10, 100, 11, 2`).

## Запуск в Kaggle

1. [Create → Notebook](https://www.kaggle.com/code), затем `File → Import Notebook`
   и загрузить `mashtots_kaggle.ipynb`.
2. `Add Input` → нужное соревнование.
3. `Settings → Accelerator: GPU T4 x2`. На CPU прогон будет очень долгим.
4. `Save & Run All (Commit)`. После завершения `submission.csv` появится в
   Output, оттуда — `Submit`.

Каталог с классами и тестовая часть ищутся автоматически, поэтому подходят оба
соревнования и любая вложенность (`Train/` или `Train/Train/`). Первая ячейка
печатает содержимое `/kaggle/input` и колонки всех CSV — если автопоиск не
сработал, из её вывода видно, какой путь подставить.

Через CLI (нужны `pip install kaggle` и `~/.kaggle/kaggle.json`):

```bash
# в kernel-metadata.json подставить свой ник Kaggle в поле "id";
# для второго соревнования заменить competition_sources на mashtots-dataset-v2
kaggle kernels push -p .
```

## Локальный запуск

```bash
pip install -r requirements.txt
kaggle competitions download -c mashtots-dataset -p data/mashtots
unzip data/mashtots/mashtots-dataset.zip -d data/mashtots
jupyter notebook mashtots_local.ipynb
```

Ожидаемая раскладка данных:

```
data/mashtots/
  Train/0/1.png ... Train/77/*.png    # 78 папок-классов
  new_test/ либо new_test.csv         # тест соревнования, опционально
  sample_submission.csv               # опционально
```

Параметры `mashtots_local.ipynb` переопределяются переменными окружения:

| Переменная | По умолчанию | Назначение |
|---|---|---|
| `MASHTOTS_DATA` | `data/mashtots` | корень датасета |
| `MASHTOTS_EPOCHS` | `30` | максимум эпох, есть early stopping |
| `MASHTOTS_MAX_PER_CLASS` | все | ограничить число изображений на класс |

## Что можно улучшить

* `REFIT_ON_ALL = True` в Kaggle-ноутбуке — дообучение на всех данных даёт ещё
  20 % обучающих изображений;
* ансамбль 3–5 прогонов с разными `SEED` и усреднением вероятностей;
* резидуальные блоки вместо простых `Conv-BN-ReLU`: на 70 тыс. изображений
  более глубокая сеть уже окупается;
* `label_smoothing=0.05` — в рукописном тексте есть объективно неоднозначные
  образцы, и смягчение метки снижает переуверенность.
