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

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.image_label)
        self.layout.addWidget(self.upload_button)
        self.layout.addWidget(self.segment_button)

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

    def segment_image(self):
        if self.image is None:
            return

        # Преобразование изображения в оттенки серого
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Применение алгоритма сегментации (например, Canny)
        edges = cv2.Canny(gray_image, 100, 200)

        # Нахождение контуров
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        # Создание копии изображения для рисования контуров
        contour_image = np.zeros_like(self.image)
        contour_image[:] = (255, 255, 255)

        # Рисование контуров на изображении
        cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

        # Отображение изображения с контурами
        self.image = contour_image
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
