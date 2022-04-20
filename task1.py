"""
Задача: Известно, что сульфид имеет длину в два раза больше чем ширину.
На изображении есть сульфиды и оксиды, задетектируйте обе и выделите контуры разными цветами.
"""

import matplotlib.pyplot as plt
import cv2
import numpy as np


def thresh_by_color(contours):
    sulfide_contours, oxide_contours = [], []
    print(len(contours))
    for i in range(len(contours)):
        width, heigth = cv2.minAreaRect(contours[i])[1]
        if 2 * min(width, heigth) > max(width, heigth):
            sulfide_contours.append(contours[i])
        else:
            oxide_contours.append(contours[i])
    return sulfide_contours, oxide_contours

def areas(contours):
    return np.array([cv2.contourArea(contour) for contour in contours])

# Считываем изображение
path = r'/home/grigoriy/Data science/Tasks/Task1/bacteria.jpg'
src = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
img = src.copy()

# 1) Устанавливаем адаптивный порог, который равен свертке изображения и окна Гаусса
# Или взвешенной сумме пикселей Гауссовым окном с учетом вычета C
thresh = cv2.adaptiveThreshold(img,  # Исходное изображение
                               255,  # Максимальное значение, которое обрабатывается адаптивным порогом
                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # Метод взвешивания: окно Гаусса
                               cv2.THRESH_BINARY,  # Порог двоичный(две области деления)
                               15,  # ширина/длина квадратного ядра Гаусса
                               10)  # C: Вычитаемая константа из взвешенной суммы

# 2) Извлекаем контуры
contours, hierarchy = cv2.findContours(thresh,  # Изображение с решающего устройства
                                       cv2.RETR_LIST,  # Иерархия порогов - дерево из их уровней вложенности
                                       cv2.CHAIN_APPROX_SIMPLE)  # Пороги аппроксимируются прямыми и хранятся в виде угловых точек

# 3) Поскольку лишних два и более контура, выполним сортировку:
contours_sorted = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)[2::]
# 4) Grayscale -> RGB
img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
# 5) Группируем на сульфиды и оксиды
sulfide_contours, oxide_contours = thresh_by_color(contours_sorted)
# 6) Найдем площади
sulfide_areas = areas(sulfide_contours)
oxide_areas = areas(oxide_contours)
# 7) Добавляем на картинку
img = cv2.drawContours(img, oxide_contours, -1, (255, 0, 0), 2)
img = cv2.drawContours(img, sulfide_contours, -1, (0, 0, 255), 2)

plt.figure()
plt.imshow(img)
plt.show()

#6) Выведем площадь, количество сульфидов и оксидов
print(f'Сульфиды: \n'
      f'Количество: {len(sulfide_contours)}\n'
      f'Площади: {sulfide_areas}')
print(f'Оксиды: \n'
      f'Количество: {len(sulfide_contours)}\n'
      f'Площади: {oxide_areas}')


