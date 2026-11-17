# Blood Cell Detection — YOLOv3 from Scratch

**Object Detection · PyTorch · YOLOv3 · CSP-Darknet · PAN-FPN · CIoU · Focal Loss**

## Overview

Проект посвящён детекции клеток крови на микроскопических изображениях. Реализован детектор YOLOv3 на **PyTorch from scratch**, затем исследовано влияние замены backbone, добавления PAN-FPN neck и модификации функции потерь.

**Задача:** локализовать и классифицировать три типа объектов — **WBC** (лейкоциты), **RBC** (эритроциты) и **Platelets** (тромбоциты).

**Датасет:** [Blood Cell Detection Dataset — Kaggle](https://www.kaggle.com/datasets/adhoppin/blood-cell-detection-datatset/data).

## From-scratch implementation

В ноутбуке реализованы собственные компоненты архитектуры и пайплайна детекции:
- YOLOv3 с Darknet-53, residual blocks и multi-scale detection heads;
- CSP-Darknet backbone: **Conv, Bottleneck, C2f, SPPF**;
- **PAN-FPN** neck для агрегации multi-scale признаков;
- **CIoU** и **Focal BCE** loss;
- декодирование bounding boxes, NMS, настройка confidence/IoU thresholds;
- подготовка данных, anchor clustering через K-means, обучение и оценка.

Проект не использует готовый YOLO-детектор из Ultralytics.

## Experimental setup

Последовательно сравниваются четыре конфигурации на одинаковом наборе данных. Пороги confidence и NMS IoU подбираются на validation. Качество и скорость инференса оцениваются на test.

**Основная метрика:** mAP@0.50:0.95. Дополнительные метрики: mAP@0.50, mAP@0.75, время инференса на изображение.

## Results

| Configuration | Test mAP@0.50:0.95 | Test mAP@0.50 | Test mAP@0.75 | Inference (ms/image) |
|---|---:|---:|---:|---:|
| YOLOv3 / Darknet-53 | 0.1668 | 0.5834 | 0.0316 | 68.46 |
| + CSP-Darknet | 0.2208 | 0.5993 | 0.0472 | 238.72 |
| + PAN-FPN | 0.1582 | 0.4896 | 0.0415 | 680.06 |
| + CIoU / Focal Loss | **0.4302** | **0.8053** | **0.4588** | **36.33** |

Все показатели взяты из **сохранённых результатов выполнения** исходного ноутбука. Наиболее высокое качество детекции получено в финальной конфигурации.

## Repository structure

```text
.
├── blood_cell_detection_yolo_from_scratch.ipynb
└── README.md
```

## Tech stack

Python, PyTorch, torchvision, NumPy, matplotlib, scikit-learn (K-means), torchmetrics, PIL.

## Notebook contents

1. Данные и визуализация bounding boxes
2. YOLOv3 / Darknet-53 from scratch
3. CSP-Darknet backbone
4. PAN-FPN neck
5. CIoU + Focal loss
6. Сравнение качества и визуализация предсказаний

**Сохранённые результаты и графики доступны непосредственно в ноутбуке без повторного обучения.**

## Dataset and reproduction

Для воспроизведения экспериментов загрузите [Blood Cell Detection Dataset](https://www.kaggle.com/datasets/adhoppin/blood-cell-detection-datatset/data), подготовьте структуру `train/`, `valid/`, `test/` с папками `images/` и `labels/`, укажите локальные пути `DATA_ROOT` и `path_to_dir` в ноутбуке. Для обучения рекомендуется CUDA GPU.
