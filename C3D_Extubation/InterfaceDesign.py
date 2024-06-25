import sys
import DisplayUI
from PyQt5.QtWidgets import QApplication, QMainWindow
from VideoDisplay import Display


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWnd = QMainWindow()
    ui = DisplayUI.Ui_UE_Surveillance()
    # 可以理解成将创建的ui绑定到新建的mainWnd上
    ui.setupUi(mainWnd)
    display = Display(ui, mainWnd)
    mainWnd.show()

    sys.exit(app.exec_())







