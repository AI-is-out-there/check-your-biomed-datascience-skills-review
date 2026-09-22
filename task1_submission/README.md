# Task 1 — Classification Solution

## Описание
Нейросеть FFN на PyTorch (2 скрытых слоя, ReLU, Dropout).
Бинарная классификация: 0 или 1.

## Результаты на валидации
- Accuracy:  0.9375
- Precision: 0.8800
- Recall:    0.7586
- F1-score:  0.8148
- ROC-AUC:   0.9834

## Как воспроизвести
1. pip install torch pandas numpy scikit-learn matplotlib
2. Положить train.csv и test_features.csv в /content/
3. Запустить your-sollution/task1_solution.ipynb
4. Результат: outputs/task1_predictions.csv

## Файлы
- your-sollution/task1_solution.ipynb — код решения
- outputs/task1_predictions.csv — предсказания