import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton
from PyQt5.QtWidgets import QVBoxLayout, QWidget, QFileDialog
from PyQt5.QtGui import QPixmap, QImage
import cv2
import numpy as np
import os
from PyQt5.QtCore import QCoreApplication
from PyQt5.QtCore import Qt


class ImageSegmentationApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('Segmentation App')
        self.setGeometry(100, 100, 1200, 800)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)

        self.segment_button = QPushButton('Segment Image')
        self.segment_button.clicked.connect(self.segment_image)

        self.upload_button = QPushButton('Upload Image')
        self.upload_button.clicked.connect(self.upload_image)

        self.increase_red_contrast_button = QPushButton(
            'Increase Red Contrast')
        self.increase_red_contrast_button.clicked.connect(
            self.red_channel_contrast)

        self.increase_green_contrast_button = QPushButton(
            'Increase Green Contrast')
        self.increase_green_contrast_button.clicked.connect(
            self.green_channel_contrast)

        self.increase_blue_contrast_button = QPushButton(
            'Increase Blue Contrast')
        self.increase_blue_contrast_button.clicked.connect(
            self.blue_channel_contrast)

        self.blank_background_button = QPushButton(
            'Blank Background')
        self.blank_background_button.clicked.connect(
            self.blank_background)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.image_label)
        self.layout.addWidget(self.upload_button)
        self.layout.addWidget(self.segment_button)
        self.layout.addWidget(self.increase_red_contrast_button)
        self.layout.addWidget(self.increase_green_contrast_button)
        self.layout.addWidget(self.increase_blue_contrast_button)
        self.layout.addWidget(self.blank_background_button)

        self.central_widget = QWidget()
        self.central_widget.setLayout(self.layout)
        self.setCentralWidget(self.central_widget)

        self.image = None

    def upload_image(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, 'Open Image', '.', 'Image Files (*.png *.jpg *.jpeg *.bmp)')
        if filename:
            self.image = cv2.imread(filename)
            self.display_image()

    def display_image(self):
        height, width, channel = self.image.shape
        bytes_per_line = 3 * width
        q_img = QImage(self.image.data, width, height,
                       bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap(q_img)
        self.image_label.setPixmap(pixmap.scaled(
            self.image_label.size(), Qt.KeepAspectRatio))

    def green_channel_contrast(self):
        if self.image is None:
            return

        # Разделение изображения на каналы
        b, g, r = cv2.split(self.image)

        # Увеличение контраста зеленого канала
        g = cv2.equalizeHist(g)

        # Объединение каналов
        self.image = cv2.merge((b, g, r))

        self.display_image()

    def red_channel_contrast(self):
        if self.image is None:
            return

        # Разделение изображения на каналы
        b, g, r = cv2.split(self.image)

        # Увеличение контраста красного канала
        r = cv2.equalizeHist(r)

        # Объединение каналов
        self.image = cv2.merge((b, g, r))

        self.display_image()

    def blue_channel_contrast(self):
        if self.image is None:
            return

        # Разделение изображения на каналы
        b, g, r = cv2.split(self.image)

        # Увеличение контраста синего канала
        b = cv2.equalizeHist(b)

        # Объединение каналов
        self.image = cv2.merge((b, g, r))

        self.display_image()

    def calc_countour_center(self, contour):
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            return cX, cY
        return None

    def find_points_coords(self, picture):
        # segmentation
        edges = cv2.Canny(picture, 100, 200)
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        points = []
        for contour in contours:
            # check if countour is circle
            approx = cv2.approxPolyDP(
                contour, 0.03 * cv2.arcLength(contour, True), True)
            if len(approx) >= 10:
                cX, cY = self.calc_countour_center(contour)
                if cX is not None:
                    points.append([cX, cY])
        return points

    def check_point_in_contour(self, point, contour):
        return cv2.pointPolygonTest(contour, (point[0], point[1]), False) > 0

    def find_closest_points(self, support_points, points, contours):
        # calculate how many points are the closest for each support point
        # create list of the same size as support points
        # each element of the list is a list of indexes of the closest points
        closest_points = [0 for _ in range(len(support_points))]
        for point in points:
            min_support_points_idx = 0
            min_dist = np.linalg.norm(
                np.array(support_points[0]) - np.array(point))
            for i, support_point in enumerate(support_points):
                # check if point is in contour
                if not self.check_point_in_contour(point, contours[i]):
                    continue
                dist = np.linalg.norm(
                    np.array(support_point) - np.array(point))
                if dist < min_dist:
                    min_dist = dist
                    min_support_points_idx = i

            closest_points[min_support_points_idx] += 1

        return closest_points

    # Функция для усиления контраста

    def increase_contrast(self, channel):
        # Применение гистограммного преобразования
        channel = cv2.equalizeHist(channel)
        return channel

    def segment_image(self):
        if self.image is None:
            return

        red_channel = self.image[:, :, 2]
        green_channel = self.image[:, :, 1]
        blue_channel = self.image[:, :, 0]

        red_channel = self.increase_contrast(red_channel)
        green_channel = self.increase_contrast(green_channel)
        blue_channel = self.increase_contrast(blue_channel)

        # Преобразование изображения в оттенки серого
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_image, 100, 200)

        # Нахождение контуров
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Создание копии изображения для рисования контуров и центров
        result_image = self.image.copy()

        # Поиск треугольных контуров
        number_of_triangles = 0
        edge_points = []
        edge_contours = []

        for contour in contours:
            # Аппроксимация контура полигоном
            approx = cv2.approxPolyDP(
                contour, 0.03 * cv2.arcLength(contour, True), True)
            # Проверка, является ли полигон треугольником (три вершины)
            if len(approx) == 3:
                number_of_triangles += 1
                # Рисование контура на изображении
                cv2.drawContours(result_image, [contour], -1, (0, 255, 0), 2)

                # Вычиисление вершин трейгольника
                for point in approx:
                    x, y = point[0]
                    edge_points.append([x, y])
                    edge_contours.append(contour)

                # Вычисление центра масс контура
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    # Рисование центра треугольника
                    cv2.circle(result_image, (cX, cY), 5, (255, 0, 0), -1)
                    # Подпись координат центра на изображении
                    cv2.putText(result_image, f"({cX}, {cY})", (cX - 50, cY - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Отображение изображения с контурами треугольников и центрами

        # find points
        points_red = self.find_points_coords(red_channel)
        points_green = self.find_points_coords(green_channel)
        points_blue = self.find_points_coords(blue_channel)

        points = points_red + points_green + points_blue

        # find support points
        num_closest = self.find_closest_points(
            edge_points, points, edge_contours)

        for i, point in enumerate(edge_points):
            # draw number of closest points
            cv2.putText(result_image, f"{num_closest[i]}", (point[0], point[1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        self.image = result_image
        self.display_image()

    def blank_background(self):
        if self.image is None:
            return

        # Преобразование изображения в оттенки серого
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_image, 100, 200)

        # Нахождение контуров
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        # Создание маски для фона
        mask = np.zeros_like(gray_image)

        # Создание копии изображения для рисования контуров
        contour_image = self.image.copy()

        # Поиск треугольных контуров и создание маски для них
        for contour in contours:
            # Аппроксимация контура полигоном
            approx = cv2.approxPolyDP(
                contour, 0.03 * cv2.arcLength(contour, True), True)
            # Проверка, является ли полигон треугольником (три вершины)
            if len(approx) == 3:
                # Рисование контура на изображении
                cv2.drawContours(contour_image, [contour], -1, (0, 255, 0), 2)
                # Создание маски для треугольных контуров
                cv2.drawContours(mask, [contour], -1, 255, -1)

        # Замена фона на белый цвет везде, кроме областей,
        #  найденных треугольников
        background = cv2.bitwise_not(mask)
        result_image = cv2.bitwise_and(cv2.merge(
            [background]*3), contour_image) + \
            cv2.bitwise_and(cv2.merge([mask] * 3), self.image)

        # Отображение изображения с контурами треугольников
        self.image = result_image
        self.display_image()


plugin_path = QCoreApplication.libraryPaths()[0]

if __name__ == '__main__':
    print(plugin_path)
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path
    os.environ['QT_PLUGIN_PATH'] = plugin_path
    print("Initilizing app...")
    app = QApplication(sys.argv)
    print("App initialized")
    window = ImageSegmentationApp()
    window.show()
    sys.exit(app.exec_())
