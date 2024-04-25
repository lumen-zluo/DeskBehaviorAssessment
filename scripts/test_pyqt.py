import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QDialog, QPushButton, QLabel, QVBoxLayout
from PyQt5.QtCore import Qt


class SubWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("全屏子窗口")
        self.setGeometry(100, 100, 400, 300)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)

        label = QLabel("字", self)
        label.setStyleSheet("font-size: 100px;")

        layout.addWidget(label)

        button = QPushButton("全屏", self)
        button.clicked.connect(self.toggle_fullscreen)
        layout.addWidget(button)

    def toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("主窗口")
        self.setGeometry(100, 100, 800, 600)

        button = QPushButton("打开子窗口", self)
        button.clicked.connect(self.open_subwindow)

    def open_subwindow(self):
        subwindow = SubWindow(self)
        subwindow.show()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())