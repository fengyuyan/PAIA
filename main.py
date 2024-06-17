import sys
from PyQt5.QtWidgets import QApplication
from frontend.mainwindow import MainWindow
from qt_material import apply_stylesheet


if __name__ == '__main__':
    app = QApplication(sys.argv)
    # apply_stylesheet(app, theme='light_blue.xml')
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())