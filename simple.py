import numpy as np


# Функция сигмоида
# Необходима для опредления значения весов
def sigmoid(x, der=False):
    if der:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))


# Набор входных данных
x = np.array([[1, 0, 1],
        [1, 0, 1],
        [0, 1, 0],
        [0, 1, 0]])

# Выходные данные
y = np.array([[0, 0, 1, 1]]).T

# Сделаем случайные числа более определёнными
np.random.seed(1)

# Инициализируем веса случайным образом со средним 0
syn0 = 2 * np.random.random((3, 1)) - 1

l1 = []

for iter in range(10000):
    # Прямое распространение
    l0 = x
    l1 = sigmoid(np.dot(l0, syn0))

    # Насколько мы ошиблись?
    l1_error = y - l1

    # Перемножим это с наклоном сигмоиды
    # на основе значений в l1
    l1_delta = l1_error * sigmoid(l1, True)

    # Обновим веса
    syn0 += np.dot(l0.T, l1_delta)

print("Выходные данные после тренеровки:")
print(l1)