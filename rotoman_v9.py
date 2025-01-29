from threading import *
from PIL import Image
import numpy as np
import re
import cv2
import sys
from PyQt5 import QtWidgets as qtw
from PyQt5 import QtCore as qtc
from PyQt5 import QtGui as qtg
import torch
import torchvision.transforms as T
from torchvision import models
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

fcn = None


class RotoMan(qtw.QMainWindow):
    def __init__(self):
        super(RotoMan, self).__init__()
        self.setWindowTitle("ROTOMAN")
        self.resize(1920, 1200)
        self.show()
        self.setStyleSheet("background-color: #070710 ; color: white;")
        self.bg = "background-color: #17203c ;border-style: outset;border-radius: 04px;"
        self.bg2 = "background-color: #9a83c5 ;border-style: outset;border-radius: 04px; color: black; padding: 6px; "
        self.bg3 = "background-color: #191933 ;border-style: outset;border-radius: 04px; color: black; padding: 2px; "

        self.current_file = "/dd/dept/pipeline/Balaji/rotoArt/roto.jpeg"
        self.centralwidget = qtw.QWidget(self)
        self.setCentralWidget(self.centralwidget)
        self.label = qtw.QLabel(self)
        self.label.setCursor(qtg.QCursor(qtc.Qt.CrossCursor))
        self.label.setMaximumSize(1920, 950)
        self.label.setStyleSheet("background-color: #0b0e18 ; color: white;")

        self.line = qtw.QFrame(self)
        self.line.setFrameShape(qtw.QFrame.HLine)
        self.line.setFrameShadow(qtw.QFrame.Sunken)

        self.menu_items()
        self.frames()
        self.render_tab()
        self.filters()
        self.info_bar()
        self.app_layout()
        self.actions()
        self.defaults()
        self.loadImage()


    def actions(self):

        self.actionOpen_Image.triggered.connect(self.open_image)
        self.actionOpen_Directory.triggered.connect(self.open_directory)
        self.actionOpen_Directory.setShortcut(qtg.QKeySequence("Ctrl+O"))
        self.actionQuit.triggered.connect(self.exitApp)
        self.actionQuit.setShortcut(qtg.QKeySequence("Ctrl+Q"))
        self.actionStartAgain.triggered.connect(self.startAgain)
        self.actionStartAgain.setShortcut(qtg.QKeySequence("Ctrl+S"))
        self.left_frame.setShortcut(qtg.QKeySequence(qtc.Qt.Key_Left))
        self.right_frame.setShortcut(qtg.QKeySequence(qtc.Qt.Key_Right))
        self.right_frame.clicked.connect(self.next_image)
        self.right_play.clicked.connect(self.next_play)
        self.left_frame.clicked.connect(self.previous_image)
        self.left_play.clicked.connect(self.previous_play)
        self.render_click.clicked.connect(self.thread)
        self.reset.clicked.connect(self.reset_view)
        self.button_zoom_in.clicked.connect(self.on_zoom_in)
        self.button_fit.clicked.connect(self.fitToFrame)
        self.button_zoom_out.clicked.connect(self.on_zoom_out)

    def info_bar(self):
        self.info = qtw.QLabel(self)
        self.info.setMaximumHeight(20)
        self.info.setText("info")
        self.info_layout = qtw.QHBoxLayout()
        self.info_layout.addWidget(self.info)

    def filters(self):

        self.gain_slider = qtw.QSlider(self)
        self.gain_slider.setOrientation(qtc.Qt.Horizontal)
        self.gain_slider.setStyleSheet(self.bg3)
        self.contrast_slider = qtw.QSlider(self)
        self.contrast_slider.setOrientation(qtc.Qt.Horizontal)
        self.contrast_slider.setStyleSheet(self.bg3)
        self.gamma_slider = qtw.QSlider(self)
        self.gamma_slider.setOrientation(qtc.Qt.Horizontal)
        self.gamma_slider.setStyleSheet(self.bg3)
        self.blur_slider = qtw.QSlider(self)
        self.blur_slider.setOrientation(qtc.Qt.Horizontal)
        self.blur_slider.setStyleSheet(self.bg3)
        self.denoise_slider = qtw.QSlider(self)
        self.denoise_slider.setOrientation(qtc.Qt.Horizontal)
        self.denoise_slider.setStyleSheet(self.bg3)
        self.brightness_slider = qtw.QSlider(self)
        self.brightness_slider.setOrientation(qtc.Qt.Horizontal)
        self.brightness_slider.setStyleSheet(self.bg3)

        self.gamma = qtw.QLabel(self)
        self.gamma.setText("gamma")
        self.blur = qtw.QLabel(self)
        self.blur.setText("blur")
        self.contrast = qtw.QLabel(self)
        self.contrast.setText("contrast")
        self.gain = qtw.QLabel(self)
        self.gain.setText("gain")
        self.denoise = qtw.QLabel(self)
        self.denoise.setText("denoise")
        self.brighness = qtw.QLabel(self)
        self.brighness.setText("brighness")

        self.filters_Layout = qtw.QGridLayout()
        self.filters_Layout.addWidget(self.gain, 0, 0, 1, 1)
        self.filters_Layout.addWidget(self.gain_slider, 0, 1, 1, 1)
        self.filters_Layout.addWidget(self.gamma, 0, 2, 1, 1)
        self.filters_Layout.addWidget(self.gamma_slider, 0, 3, 1, 1)
        self.filters_Layout.addWidget(self.blur, 0, 4, 1, 1)
        self.filters_Layout.addWidget(self.blur_slider, 0, 5, 1, 1)
        self.filters_Layout.addWidget(self.contrast, 0, 6, 1, 1)
        self.filters_Layout.addWidget(self.contrast_slider, 0, 7, 1, 1)
        self.filters_Layout.addWidget(self.denoise, 0, 8, 1, 1)
        self.filters_Layout.addWidget(self.denoise_slider, 0, 9, 1, 1)
        self.filters_Layout.addWidget(self.brighness, 0, 10, 1, 1)
        self.filters_Layout.addWidget(self.brightness_slider, 0, 11, 1, 1)

        # Blur slider

        self.blur_slider.setMinimum(1)
        self.blur_slider.setMaximum(50)
        self.blur_slider.setValue(1)
        self.blur_slider.setTickPosition(qtw.QSlider.TicksBelow)
        self.blur_slider.setTickInterval(10)
        self.blur_slider.valueChanged.connect(self.loadImage)

        # contrast slider

        self.contrast_slider.setMinimum(0)
        self.contrast_slider.setMaximum(3)
        self.contrast_slider.setValue(1)
        self.contrast_slider.setTickPosition(qtw.QSlider.TicksBelow)
        self.contrast_slider.setTickInterval(1)
        self.contrast_slider.valueChanged.connect(self.loadImage)

        self.brightness_slider.setMinimum(-127)
        self.brightness_slider.setMaximum(127)
        self.brightness_slider.setValue(0)
        self.brightness_slider.setTickPosition(qtw.QSlider.TicksBelow)
        self.brightness_slider.setTickInterval(30)
        self.brightness_slider.valueChanged.connect(self.loadImage)


    def frames(self):

        self.in_label = qtw.QLabel(self)
        self.in_label.setText("in")
        self.inp = qtw.QLineEdit(self)
        self.inp.setStyleSheet(self.bg)
        self.fframe_display = qtw.QLCDNumber(self)
        self.left_frame = qtw.QPushButton(self)
        self.left_frame.setShortcut("<")
        self.left_frame.setText("<")
        self.left_frame.setStyleSheet(self.bg2)
        self.left_frame.setMinimumSize(200, 20)
        self.left_play = qtw.QPushButton(self)
        self.left_play.setText("<<")
        self.left_play.setStyleSheet(self.bg2)
        self.left_play.setMinimumSize(200, 20)
        self.frame_display = qtw.QLCDNumber(self)
        self.right_frame = qtw.QPushButton(self)
        self.right_frame.setShortcut(">")
        self.right_frame.setText(">")
        self.right_frame.setStyleSheet(self.bg2)
        self.right_frame.setMinimumSize(200, 20)
        self.right_play = qtw.QPushButton(self)
        self.right_play.setText(">>")
        self.right_play.setStyleSheet(self.bg2)
        self.right_play.setMinimumSize(200, 20)
        self.lframe_display = qtw.QLCDNumber(self)
        self.out = qtw.QLineEdit(self)
        self.out.setStyleSheet(self.bg)
        self.out_label = qtw.QLabel(self)
        self.out_label.setText("out")
        self.reset = qtw.QPushButton(self)
        self.reset.setText("RESET")
        self.reset.setStyleSheet(self.bg2)
        self.reset.setMinimumSize(200, 20)
        self.button_zoom_in = qtw.QPushButton(self)
        self.button_zoom_in.setText('Zoom +')
        self.button_zoom_in.setShortcut("+")
        self.button_zoom_in.setStyleSheet(self.bg2)
        self.button_fit = qtw.QPushButton(self)
        self.button_fit.setText('Fit to Frame')
        self.button_fit.setShortcut("F")
        self.button_fit.setStyleSheet(self.bg2)
        self.button_zoom_out = qtw.QPushButton(self)
        self.button_zoom_out.setText('Zoom -')
        self.button_zoom_out.setShortcut("-")
        self.button_zoom_out.setStyleSheet(self.bg2)

        self.frames_layout = qtw.QHBoxLayout()
        self.frames_layout.addWidget(self.in_label)
        self.frames_layout.addWidget(self.inp)
        self.frames_layout.addWidget(self.fframe_display)
        self.frames_layout.addWidget(self.left_frame)
        self.frames_layout.addWidget(self.frame_display)
        self.frames_layout.addWidget(self.right_frame)
        self.frames_layout.addWidget(self.lframe_display)
        self.frames_layout.addWidget(self.out)
        self.frames_layout.addWidget(self.out_label)
        self.frames_layout.addWidget(self.reset)
        self.frames_layout.addWidget(self.button_zoom_out)
        self.frames_layout.addWidget(self.button_fit)
        self.frames_layout.addWidget(self.button_zoom_in)
        
    def render_tab(self):

        self.location = qtw.QLabel(self)
        self.location.setText("Save Location")
        self.location_value = qtw.QLineEdit(self)
        self.location_value.setStyleSheet(self.bg)
        self.heightt = qtw.QLabel(self)
        self.heightt.setText("Height")
        self.height_value = qtw.QLineEdit(self)
        self.height_value.setStyleSheet(self.bg)
        self.render_click = qtw.QPushButton(self)
        self.render_click.setText("RENDER")
        self.render_click.setShortcut(qtg.QKeySequence("Ctrl+R"))
        self.render_click.setStyleSheet(self.bg2)
        self.render_click.setMinimumSize(200, 30)

        self.render_layout = qtw.QHBoxLayout()
        self.render_layout.addWidget(self.location)
        self.render_layout.addWidget(self.location_value)
        self.render_layout.addWidget(self.heightt)
        self.render_layout.addWidget(self.height_value)
        self.render_layout.addWidget(self.render_click)

    def menu_items(self):

        self.menubar = qtw.QMenuBar(self)
        self.menubar.setMaximumSize(1920, 50)
        self.menubar.setGeometry(qtc.QRect(0, 0, 1920, 50))
        self.setMenuBar(self.menubar)
        self.menufile = qtw.QMenu(self.menubar)
        self.menufile.setTitle("File")
        self.actionOpen_Image = qtw.QAction(self)
        self.actionOpen_Image.setText("Open Image")
        self.actionOpen_Directory = qtw.QAction(self)
        self.actionOpen_Directory.setText("Open Directory")
        self.actionQuit = qtw.QAction(self)
        self.actionQuit.setText("Quit")
        self.actionQuit.setShortcut("q")
        self.actionStartAgain = qtw.QAction(self)
        self.actionStartAgain.setText("startAgain")

        self.menufile.addAction(self.actionOpen_Image)
        self.menufile.addAction(self.actionOpen_Directory)
        self.menufile.addAction(self.actionQuit)
        self.menufile.addAction(self.actionStartAgain)
        self.menubar.addAction(self.menufile.menuAction())

    def app_layout(self):

        self.verticalLayout_2 = qtw.QVBoxLayout(self.centralwidget)
        self.verticalLayout_2.addWidget(self.label)
        self.verticalLayout_2.addWidget(self.line)
        self.verticalLayout_2.addLayout(self.frames_layout)
        self.verticalLayout_2.addLayout(self.render_layout)
        self.verticalLayout_2.addLayout(self.filters_Layout)
        self.verticalLayout_2.addLayout(self.info_layout)

    def loadImage(self):
        self.theimg = cv2.imread(
            self.current_file, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        self.theimg = cv2.convertScaleAbs(
            self.theimg, alpha=self.contrast_slider.value(), beta=self.brightness_slider.value())
        self.theimg = cv2.blur(
            self.theimg, (self.blur_slider.value(), self.blur_slider.value()))
        self.showimg = qtg.QImage(
            self.theimg.data, self.theimg.shape[1], self.theimg.shape[0], self.theimg.shape[1] * 3, qtg.QImage.Format_RGB888).rgbSwapped()
        self.pixmap = qtg.QPixmap.fromImage(self.showimg).scaled(self.showimg.width(
        ), self.showimg.height(), qtc.Qt.KeepAspectRatio, qtc.Qt.SmoothTransformation)
        self.label.setPixmap(self.pixmap)
        self.label.setAlignment(qtc.Qt.AlignCenter)
        self.label.resize(self.pixmap.width(), self.pixmap.height())
        self.label.setSizePolicy(
            qtw.QSizePolicy.Ignored, qtw.QSizePolicy.Ignored)

    def displayFrameNumber(self):

        self.matteNumber = re.findall('\d+', self.current_file)
        self.firstframes = (re.findall('\d+', self.file_list[0]))
        self.firstframe_number = (re.findall('\d+', self.file_list[0]))[-1]
        self.lastframe_number = (re.findall('\d+', self.file_list[-1]))[-1]
        self.fframe_display.display(self.firstframe_number)
        self.lframe_display.display(self.lastframe_number)
        self.frame_display.display(self.matteNumber[-1])
        return self.matteNumber

    def defaults(self):
        self.newimg = None
        self.file_counter = 0
        self.firstFrame = 0
        self.inputs = 0
        self.outputs = 0
        self.file_path = None
        self.file_list = []
        self.new_list = []
        self.imageHeight = 1080
        self.matteNumber = 0000
        self.firstframe_number = 0000
        self.lastframe_number = 0000
        self.defaultFframe = []
        self.savelocation = "/PATH/TO/SAVE/LOCATION" ##Update me as per your needs

    def open_image(self):

        try:
            options = qtw.QFileDialog.Options()
            selected_filter = "Images (*.png *.exr *.jpg)"
            files, _ = qtw.QFileDialog.getOpenFileName(
                self, "Open File", "", selected_filter, options=options)
            self.new_list = [os.path.basename(files)]
            self.file_path = os.path.dirname(files)
            if files != "":
                self.current_file = files
                self.loadImage()
                msg = "image width : {} , image height : {} , channels : {}  ".format(
                    self.theimg.shape[1], self.theimg.shape[0], self.theimg.shape[2])
                self.info.setText(str(msg))
                self.imageHeight = self.theimg.shape[0]
            return self.new_list, self.file_path, self.imageHeight
        except Exception as e:
            print(e)
            self.info.setText("please select the proper file")

    def open_directory(self):
        try:
            options = qtw.QFileDialog.Options()
            directory = str(
                qtw.QFileDialog.getExistingDirectory(options=options))
            self.file_path = directory
            listOffiles = os.listdir(self.file_path)
            self.new_list = listOffiles
            self.file_list = [self.file_path + "/" + f for f in sorted(
                listOffiles) if f.endswith(".exr") or f.endswith(".jpg")]
            self.defaultFframe = self.file_list[:]
            self.current_file = self.file_list[0]
            self.loadImage()
            self.displayFrameNumber()
            self.imageHeight = self.theimg.shape[0]
            self.imageWidth = self.theimg.shape[1]
            msg = "image width : {} , image height : {} , channels : {} , FirstFrame : {}, LastFrame = {} ".format(
                self.theimg.shape[1], self.theimg.shape[0], self.theimg.shape[2], self.firstframe_number, self.lastframe_number)
            self.info.setText(str(msg))
            return self.file_path, self.firstframe_number, self.lastframe_number, self.new_list, self.defaultFframe
        except Exception as e:
            print(e)
            self.info.setText(
                "please select the proper folder directory, Error details : {}".format(e))

    def sorter_func(self,x):
        num = x.split('.')[-2]
        return int(num)

    def reset_view(self):
        try:
            frame = int(re.findall('\d+', self.defaultFframe[0])[-1])
            self.inputs = int(self.inp.text())
            self.outputs = int(self.out.text())
            files_list = os.listdir(self.file_path)
            listOffiles = sorted(files_list, key=self.sorter_func)
            self.new_list = listOffiles[(
                self.inputs - frame): (self.outputs - frame)+1]
            print("new_list",self.new_list)
            self.file_list = [self.file_path + "/" + f for f in sorted(
                self.new_list) if f.endswith(".exr") or f.endswith(".jpg")]
            self.current_file = self.file_list[0]
            self.loadImage()
            self.displayFrameNumber()
            msg = "image width : {} , image height : {} , channels : {} , FirstFrame : {}, LastFrame : {} ".format(
                self.theimg.shape[1], self.theimg.shape[0], self.theimg.shape[2], self.firstframe_number, self.lastframe_number)
            self.info.setText(str(msg))
            return self.file_path, self.firstframe_number, self.lastframe_number, self.new_list
        except Exception as e:
            print(e)
            self.info.setText(
                "Please check the values provided, Error details : {}".format(e))

    def next_image(self):
        try:
            if self.file_counter is not None:
                self.file_counter += 1
                self.file_counter %= len(self.file_list)
                self.current_file = self.file_list[self.file_counter]
                self.loadImage()
                self.displayFrameNumber()
                self.resize_image()
        except Exception as e:
            print(e)
            self.info.setText("please check your folder directory again")

    def next_play(self):
        try:
            while True:
                self.next_image()
        except Exception as e:
            print(e)
            self.info.setText("please check your folder directory again")

    def previous_image(self):
        try:
            if self.file_counter is not None:
                self.file_counter -= 1
                self.file_counter %= len(self.file_list)
                self.current_file = self.file_list[self.file_counter]
                self.theimg = cv2.imread(self.current_file)
                self.loadImage()
                self.displayFrameNumber()
                self.resize_image()
        except Exception as e:
            print(e)
            self.info.setText("please check your folder directory again")

    def previous_play(self):
        self.previous_image()
        
    def on_zoom_in(self, event):
        self.imageHeight += 100
        self.resize_image()
    
    def fitToFrame(self):
        self.loadImage()
        scaled_pixmap = self.pixmap.scaled(1920,1080, qtc.Qt.KeepAspectRatio, qtc.Qt.SmoothTransformation)
        self.label.setPixmap(scaled_pixmap)
        
    def on_zoom_out(self, event):
        self.imageHeight -= 100
        self.resize_image()

    def resize_image(self):
        self.loadImage()
        scaled_pixmap = self.pixmap.scaledToHeight(self.imageHeight)
        self.label.setPixmap(scaled_pixmap)

    def getRotoModel(self):
        global fcn
        fcn = models.segmentation.deeplabv3_resnet101(pretrained=True).eval()

    def decode_segmap(self, image, nc=21):

        label_colors = np.array([(0, 0, 0),  # 0=background
                                 # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                                 (0, 0, 0), (0, 0, 0), (0, 0,
                                                        0), (0, 0, 0), (0, 0, 0),
                                 # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                                 (0, 0, 0), (0, 0, 0), (0, 0,
                                                        0), (0, 0, 0), (0, 0, 0),
                                 # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
                                 (0, 0, 0), (0, 0, 0), (0, 0,
                                                        0), (0, 0, 0), (255, 0, 0),
                                 # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
                                 (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)])

        r = np.zeros_like(image).astype(np.uint8)
        g = np.zeros_like(image).astype(np.uint8)
        b = np.zeros_like(image).astype(np.uint8)

        for l in range(0, nc):
            idx = image == l
            r[idx] = label_colors[l, 0]
            g[idx] = label_colors[l, 1]
            b[idx] = label_colors[l, 2]

        self.rgb = np.stack([r, g, b], axis=2)
        return self.rgb

    def thread(self):
        t1 = Thread(target=self.matte)
        t1.start()

    def createMatte(self, filename, matteName, size):
        img = cv2.imread(filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.blur(img, (self.blur_slider.value(),
                       self.blur_slider.value()))

        im_pil = Image.fromarray(img)

        trf = T.Compose([T.Resize(size),
                        T.ToTensor(),
                        T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])])
        inp = trf(im_pil).unsqueeze(0)
        if (fcn == None):
            self.getRotoModel()
        out = fcn(inp)['out']
        om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
        self.rgb = self.decode_segmap(om)
        self.im = Image.fromarray(self.rgb)
        self.im.save(matteName)

    def matte(self):
        try:
            folderLocation = self.file_path
            height_updated = self.height_value.text()
            self.savelocation = self.location_value.text() or self.savelocation

            for currentfile in self.new_list:
                sourceFile = folderLocation + "/" + currentfile
                mainNameend = currentfile.find(".")
                matteName = currentfile[:mainNameend] + \
                    "_matte" + currentfile[mainNameend:]
                matteDirectory = self.savelocation + "/" + matteName
                if self.imageHeight < self.imageWidth:
                    size = int(height_updated or self.imageHeight)
                else:
                    size = int(height_updated or self.imageWidth)
                self.createMatte(sourceFile, matteDirectory, size)
                sourceFile_im = cv2.imread(sourceFile)
                matteDirectory_im = cv2.imread(matteDirectory)
                if sourceFile_im.shape[0] == matteDirectory_im.shape[0]:
                    self.theimg = cv2.addWeighted(sourceFile_im, 0.5, matteDirectory_im, 0.4, 0)
                else:
                    self.theimg = matteDirectory_im
                self.showimg = qtg.QImage(
                    self.theimg.data, self.theimg.shape[1], self.theimg.shape[0], self.theimg.shape[1] * 3, qtg.QImage.Format_RGB888).rgbSwapped()
                self.pixmap = qtg.QPixmap.fromImage(self.showimg).scaled(self.showimg.width(
                ), self.showimg.height(), qtc.Qt.KeepAspectRatio, qtc.Qt.SmoothTransformation)
                self.label.setPixmap(self.pixmap)
                self.label.setAlignment(qtc.Qt.AlignCenter)
                self.label.resize(self.pixmap.width(), self.pixmap.height())
                self.label.setSizePolicy(
                    qtw.QSizePolicy.Ignored, qtw.QSizePolicy.Ignored)
                self.info.setText(matteName)
                matteNumber = re.findall('\d+', sourceFile)
                self.frame_display.display(matteNumber[-1])
                print("created", matteName)
        except Exception as e:
            print(e)
            self.info.setText("please check your values provided")

    def exitApp(self):
        qtw.QApplication.instance().quit()

    def startAgain(self):
        qtc.QCoreApplication.quit()
        status = qtc.QProcess.startDetached(sys.executable, sys.argv)
        print(status)


if __name__ == "__main__":
    app = qtw.QApplication(sys.argv)
    app.setWindowIcon(qtg.QIcon('/PATH/TO/YOUR_LOGO'))
    window = RotoMan()
    sys.exit(app.exec_())
