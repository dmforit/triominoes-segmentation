from PyQt6.QtWidgets import QApplication, QMainWindow, QScrollArea, QLabel, QVBoxLayout, QWidget, QSplitter, QPushButton
import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout,
    QHBoxLayout, QPushButton, QWidget,
    QFileDialog, QLabel, QScrollArea,
    QSplitter
)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt
from Segmentation import Segmentation
import numpy as np
import cv2


class CenteredButtonsApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Сегментация")

        self.layout = QVBoxLayout()

        self.image_label = QLabel()
        self.processed_image = QLabel()
        self.image_appeared = False
        self.image = None
        self.filename = None
        self.result_str = QLabel()
        self.result_str.setWordWrap(True)
        self.result_str.setText('')

        self.splitter = QSplitter()

        # Create a QScrollArea
        self.scroll_area = QScrollArea()
        self.scroll_area.setMinimumSize(200, 100)
        self.scroll_area.setWidgetResizable(True)

        scroll_layout = QVBoxLayout()
        scroll_layout.addWidget(self.result_str)

        scroll_widget = QWidget()
        scroll_widget.setLayout(scroll_layout)

        self.scroll_area.setWidget(scroll_widget)

        self.splitter.addWidget(self.scroll_area)

        self.import_button = QPushButton("Выбрать изображение")
        self.import_button.clicked.connect(self.import_image)
        self.import_button.setFixedSize(180, 40)

        self.process_button = QPushButton("Обработать изображение")
        self.process_button.clicked.connect(self.process_image)
        self.process_button.setFixedSize(180, 40)

        self.buttons_layout = QVBoxLayout()
        self.buttons_layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.buttons_layout.addWidget(self.process_button,
                                      alignment=Qt.AlignmentFlag.AlignCenter)
        self.buttons_layout.addWidget(self.splitter,
                                      alignment=Qt.AlignmentFlag.AlignCenter)

        self.layout.addStretch()
        self.layout.addWidget(self.import_button,
                              alignment=Qt.AlignmentFlag.AlignCenter)
        self.layout.addStretch()

        self.images_layout = QHBoxLayout()
        self.images_layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.images_layout.addWidget(self.image_label)

        widget = QWidget()
        widget.setLayout(self.layout)
        self.setCentralWidget(widget)

    def process_image(self):
        if self.filename is None or self.filename == '':
            return

        result, segmentation = Segmentation(self.filename).process()
        segmentation = cv2.cvtColor(segmentation, cv2.COLOR_BGR2RGB)

        self.result_str.setText(result)

        height, width, channels = segmentation.shape
        bytes_per_line = channels * width
        image = QImage(segmentation.data, width, height,
                       bytes_per_line, QImage.Format.Format_RGB888)

        pixmap = QPixmap.fromImage(image).scaled(700, 600)

        self.processed_image.setPixmap(pixmap)
        self.processed_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.images_layout.addWidget(self.processed_image)
        self.processed_image.show()

    def import_image(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open Image File", "", "Images (*.png *.jpg *.jpeg *.bmp *.gif)")

        if filename and filename != '':
            self.filename = filename
            self.image = cv2.imread(self.filename)

            pixmap = QPixmap(self.filename).scaled(700, 600)
            self.image_label.setPixmap(pixmap)
            self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

            if self.layout.count() == 6:
                self.result_str.setText('')

            if not self.image_appeared:
                self.layout.insertLayout(0, self.images_layout)
                self.layout.insertLayout(
                    self.layout.count() - 1, self.buttons_layout)
                self.image_appeared = True
            elif self.images_layout.count() == 2:
                self.images_layout.removeItem(self.images_layout.itemAt(1))
                self.processed_image.hide()


def main():
    app = QApplication(sys.argv)
    window = CenteredButtonsApp()
    window.resize(400, 300)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
