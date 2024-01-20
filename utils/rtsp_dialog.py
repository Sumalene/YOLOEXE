from PySide6 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(783, 40)
        Form.setMinimumSize(QtCore.QSize(0, 40))
        Form.setMaximumSize(QtCore.QSize(16777215, 41))
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/img/None.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        Form.setWindowIcon(icon)
        Form.setStyleSheet("#Form{background:rgba(120,120,120,255)}")
        self.horizontalLayout = QtWidgets.QHBoxLayout(Form)
        self.horizontalLayout.setContentsMargins(-1, 5, -1, 5)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(Form)
        self.label.setMinimumSize(QtCore.QSize(0, 30))
        self.label.setMaximumSize(QtCore.QSize(16777215, 30))
        self.label.setStyleSheet("QLabel{font-family: \"Microsoft YaHei\";\n"
                                 "font-size: 18px;\n"
                                 "font-weight: bold;\n"
                                 "color:white;}")
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.rtspEdit = QtWidgets.QLineEdit(Form)
        self.rtspEdit.setMinimumSize(QtCore.QSize(0, 31))
        self.rtspEdit.setStyleSheet("background-color: rgb(207, 207, 207);")
        self.rtspEdit.setObjectName("rtspEdit")
        self.horizontalLayout.addWidget(self.rtspEdit)
        self.rtspButton = QtWidgets.QPushButton(Form)
        self.rtspButton.setStyleSheet("QPushButton{font-family: \"Microsoft YaHei\";\n"
                                      "font-size: 18px;\n"
                                      "font-weight: bold;\n"
                                      "color:white;\n"
                                      "text-align: center center;\n"
                                      "padding-left: 5px;\n"
                                      "padding-right: 5px;\n"
                                      "padding-top: 4px;\n"
                                      "padding-bottom: 4px;\n"
                                      "border-style: solid;\n"
                                      "border-width: 0px;\n"
                                      "border-color: rgba(255, 255, 255, 255);\n"
                                      "border-radius: 3px;\n"
                                      "background-color: rgba(255,255,255,30);}\n"
                                      "\n"
                                      "QPushButton:focus{outline: none;}\n"
                                      "\n"
                                      "QPushButton::pressed{font-family: \"Microsoft YaHei\";\n"
                                      "                     font-size: 16px;\n"
                                      "                     font-weight: bold;\n"
                                      "                     color:rgb(200,200,200);\n"
                                      "                     text-align: center center;\n"
                                      "                     padding-left: 5px;\n"
                                      "                     padding-right: 5px;\n"
                                      "                     padding-top: 4px;\n"
                                      "                     padding-bottom: 4px;\n"
                                      "                     border-style: solid;\n"
                                      "                     border-width: 0px;\n"
                                      "                     border-color: rgba(255, 255, 255, 255);\n"
                                      "                     border-radius: 3px;\n"
                                      "                     background-color:  rgba(255,255,255,150);}\n"
                                      "\n"
                                      "QPushButton::hover {\n"
                                      "border-style: solid;\n"
                                      "border-width: 0px;\n"
                                      "border-radius: 0px;\n"
                                      "background-color: rgba(255,255,255,50);}")
        self.rtspButton.setObjectName("rtspButton")
        self.horizontalLayout.addWidget(self.rtspButton)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.label.setText(_translate("Form", "rtsp address:"))
        self.rtspButton.setText(_translate("Form", "confirm"))
# import apprcc_rc
