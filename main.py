from numpy import exp, array, random, dot


class NeuralNetwork():
    def __init__(self):
        # Проводим рандомизацию
        random.seed(1)

        # Моделируем нейрон с тремя входами и одним выходом
        # Присваиваем случайные веса матрице 3 x 1 со значениями в диапазоне -1 to 1
        self.synaptic_weights = 2 * random.random((3, 1)) - 1

    # Используем сигмоид-функцию
    # Она нужна для нормализации значений в диапазоне от 0 до 1
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    # Производная сигмоидной функции
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    # Обучение нейронной сети
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in range(number_of_training_iterations):
            # Пропускаем трен ировочный набор через сеть
            output = self.think(training_set_inputs)

            # Вычисляем ошибку
            error = training_set_outputs - output

            # Multiply the error by the input and again by the gradient of the Sigmoid curve.
            # This means less confident weights are adjusted more.
            # This means inputs, which are zero, do not cause changes to the weights.
            adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))

            # Корректируем веса
            self.synaptic_weights += adjustment

    # Основная функция работы нейронной сети
    def think(self, inputs):
        # Пропускаем "сигнал" через нейрон
        return self.__sigmoid(dot(inputs, self.synaptic_weights))


if __name__ == "__main__":

    # Инициализация сети
    neural_network = NeuralNetwork()

    print("Random starting synaptic weights: ")
    print(neural_network.synaptic_weights)

    # Тренировочный набор
    # одно выходное значение
    training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    training_set_outputs = array([[0, 1, 1, 0]]).T

    # Обучаем сеть по тренировочному набору
    # Делаем это 10000 раз, всякий раз уменьшая ошибку
    neural_network.train(training_set_inputs, training_set_outputs, 10000)

    print("New synaptic weights after training: ")
    print(neural_network.synaptic_weights)

    # Даем сети проверочное задание
    print("Considering new situation [1, 0, 0] -> ?: ")
    print(neural_network.think(array([1, 0, 0])))