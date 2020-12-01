from PyQt5 import QtWidgets
from UI import InputUI
import sys


class Build_InputUI(QtWidgets.QMainWindow):
    text='String'
    def __init__(self,LabelText='Hello World'):

        super(Build_InputUI, self).__init__()
        self.ui = InputUI.Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.label.setText(LabelText)
        self.ui.pushButton.setText('Input')
        self.ui.pushButton.clicked.connect(self.Clik_Button)

    def Clik_Button(self):
        self.text = self.ui.lineEdit.text()
        self.ui.lineEdit.clear()

    def get_text(self):

        return self.text





if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    window = Build_InputUI()

    window.show()
    sys.exit(app.exec_())
