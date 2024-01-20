import cv2
from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *


class CameraViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.capture = None
        self.setup_ui()
        self.capture_frame = None

    def setup_ui(self):
        self.setWindowTitle("Camera Viewer")
        self.resize(640, 480)

        self.image_label = QLabel(self)
        self.image_label.setMinimumSize(640, 480)
        self.image_label.setAlignment(Qt.AlignCenter)

        self.start_button = QPushButton("Start", self)
        self.start_button.clicked.connect(self.start_capture)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.image_label)
        main_layout.addWidget(self.start_button)

        self.setLayout(main_layout)

    @Slot()
    def start_capture(self):
        self.capture = cv2.VideoCapture(0)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.display_frame)
        self.timer.start(60)

    def display_frame(self):
        ret, self.capture_frame = self.capture.read()
        if ret:
            frame = cv2.cvtColor(self.capture_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            img = QImage(frame.data, w, h, ch * w, QImage.Format_RGB888)
            self.image_label.setPixmap(QPixmap.fromImage(img))

    def getcapture_frame(self):
        return self.capture_frame

if __name__ == "__main__":
    app = QApplication()
    viewer = CameraViewer()
    viewer.show()
    app.exec_()