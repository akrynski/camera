#-------------------------------------------------------------------------------
# Name:        Camera
# Purpose:     Internal class for video imaging purposes
#
# Author:      Andrzej Kryński
#
# Created:     16-06-2013
# Copyright:   (c) Andrzej Kryński 2013
# Licence:     GNU LESSER GENERAL PUBLIC LICENSE
#-------------------------------------------------------------------------------
import cv2
import cv2.cv as cv
import Tkinter as tk
from ttk import Combobox, Labelframe
from PIL import Image, ImageTk, ImageDraw
import numpy
from matplotlib import pyplot as plt
##import Image
##import ImageTk

#from Tkinter import *
#import numpy

class Camera:
    """
    ImageTk.PhotoImage(image) => PhotoImage instance
    Creates a Tkinter-compatible photo image, which can be used everywhere Tkinter
    expects an image object. If the image is an RGBA image, pixels having alpha 0
    are treated as transparent.

    create_image(x0, y0, options...) => id
    Create a image item placed relative to the given position.
    Note that the image itself is given by the image option.
    image: The image object (a Tkinter PhotoImage or BitmapImage instance,
    or instances of the corresponding Python Imaging Library classes).
    anchor: Specifies which part of the image that should be placed at the given position.
    Use one of N, NE, E, SE, S, SW, W, NW, or CENTER. Default is CENTER.

    Warning
    There is a bug in the current version of the Python Imaging Library that can cause
    your images not to display properly. When you create an object of class PhotoImage,
    the reference count for that object does not get properly incremented, so unless
    you keep a reference to that object somewhere else, the PhotoImage object may be
    garbage-collected, leaving your graphic blank on the application.
    """

    def __init__(self,cam,root,canvas,histCanvas,frame,position):
        self.cam = cam
        self.root = root
        self.canvas = canvas
        self.histCanvas = histCanvas
        self.frame = frame
        self.onoff = False
        self.detect = False
        self.effect = 'none'
        self.image = None
        self.a = None
        self.b = None
        self.contrast = 0.0
        self.brightness = 0.0
        self.position = position
        self.faceX = 0
        self.faceY = 0

##==============================================================================
    def getFaceX(self):
        return str(self.faceX)
    def getFaceY(self):
        return str(self.faceY)
##==============================================================================
    def setContrast(self, value):
        self.contrast = value
        self.cam.set(cv.CV_CAP_PROP_CONTRAST, self.contrast)
##==============================================================================
    def setBrightness(self, value):
        self.brightness = value
        self.cam.set(cv.CV_CAP_PROP_BRIGHTNESS, self.brightness)
##==============================================================================
    def setEffect(self, effect):
        self.effect = effect
##==============================================================================
    def draw_str(self, dst, (x, y), s):
        cv2.putText(dst, s, (x+2, y+2), cv2.FONT_HERSHEY_COMPLEX, 1.0,
        cv.RGB(0, 255, 0), thickness = 2, lineType=cv2.CV_AA, bottomLeftOrigin=False)
        cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_COMPLEX, 1.0,
        cv.RGB(255, 0, 0), lineType=cv2.CV_AA, bottomLeftOrigin=False)
##==============================================================================
    def drawHistogram(self,image):
        '''input of numpy.ndarray type, output is PhotoImage type'''

        h = numpy.zeros((300,256,3))
        b,g,r = image[:,:,0].copy(),image[:,:,1].copy(),image[:,:,2].copy()
        bins = numpy.arange(257)
        bin = bins[0:-1]
        color = [ (255,0,0),(0,255,0),(0,0,255) ]

        for item,col in zip([b,g,r],color):
            N,bins = numpy.histogram(item,bins)
            v=N.max()
            N = numpy.int32(numpy.around((N*255)/v))
            N=N.reshape(256,1)
            pts = numpy.column_stack((bin,N))
            cv2.polylines(h,[pts],False,col,2)

        h=numpy.flipud(h)

        #convert numpy.ndarray to iplimage
        ipl_img = cv2.cv.CreateImageHeader((h.shape[1], h.shape[0]), cv.IPL_DEPTH_8U,3)
        cv2.cv.SetData(ipl_img, h.tostring(),h.dtype.itemsize * 3 * h.shape[1])
        img = self.ipl2tk_image(ipl_img)

        return img
##==============================================================================
    def ipl2tk_image(self, iplimg):
        '''convert iplimage to PhotoImage via PIL image'''
        pil_image = Image.fromstring(
                'RGB',
                cv.GetSize(iplimg),
                iplimg.tostring(),
                'raw',
                'BGR',
                iplimg.width*3,
                0)
        return ImageTk.PhotoImage(pil_image)
##==============================================================================
    def printCamResolution(self,camera):
        print 'Camera Capture Resolution: '
        print 10*'='
        print 'Wideo width {0}, Wideo height {1}'.format(camera.get\
        (cv.CV_CAP_PROP_FRAME_WIDTH), camera.get(cv.CV_CAP_PROP_FRAME_HEIGHT))
                                                                                #using VideoCapture we'd better use get/set methods of camera instance
                                                                                #insted of cv.Get/SetCaptureProperty
        print
##==============================================================================
    def loadHaars(self):
        faceCascade = cv.Load("haarcascades/haarcascade_frontalface_alt.xml")
        eyeCascade = cv.Load("haarcascades/haarcascade_eye_tree_eyeglasses.xml")
        mouthCascade = cv.Load("haarcascades/haarcascade_mcs_mouth.xml")

        return (faceCascade, eyeCascade, mouthCascade)
##==============================================================================
    def detectFace(self, cam_img, faceCascade, eyeCascade, mouthCascade):       #cam_img should be cv2.cv.iplcam_img
        min_size = (20,20)
        image_scale = 2
        haar_scale = 1.2
        min_neighbors = 2
        haar_flags = 0
        image_width = int(cam_img.get(cv.CV_CAP_PROP_FRAME_WIDTH))
        image_height = int(cam_img.get(cv.CV_CAP_PROP_FRAME_HEIGHT))
        # Allocate the temporary images
        gray = cv.CreateImage((image_width, image_height), 8, 1)                #tuple as the first arg
        smallImage = cv.CreateImage((cv.Round(image_width / image_scale),cv.Round (image_height / image_scale)), 8 ,1)

        (ok,img) = cam_img.read()
        #print 'gray is of ',type(gray) >>> gray is of  <type 'cv2.cv.iplimage'>
        #print type(smallImage)  >>> <type 'cv2.cv.iplimage'>
        #print type(image) >>> <type 'cv2.VideoCapture'>
        #print type(img) >>> <type 'numpy.ndarray'>

        #convert numpy.ndarray to iplimage
        ipl_img = cv2.cv.CreateImageHeader((img.shape[1], img.shape[0]), cv.IPL_DEPTH_8U,3)
        cv2.cv.SetData(ipl_img, img.tostring(),img.dtype.itemsize * 3 * img.shape[1])
  

        # Convert color input image to grayscale
        cv.CvtColor(ipl_img, gray, cv.CV_BGR2GRAY)

        # Scale input image for faster processing
        cv.Resize(gray, smallImage, cv.CV_INTER_LINEAR)

        # Equalize the histogram
        cv.EqualizeHist(smallImage, smallImage)

        # Detect the faces
        faces = cv.HaarDetectObjects(smallImage, faceCascade, cv.CreateMemStorage(0),
        haar_scale, min_neighbors, haar_flags, min_size)
        # => The function returns a list of tuples, (rect, neighbors) , where rect is a CvRect specifying the object’s extents and neighbors is a number of neighbors.
        # => CvRect cvRect(int x, int y, int width, int height)
        # If faces are found
        if faces:
            face = faces[0]
            self.faceX = face[0][0]
            self.faceY = face[0][1]

            for ((x, y, w, h), n) in faces:
            # the input to cv.HaarDetectObjects was resized, so scale the
            # bounding box of each face and convert it to two CvPoints
                pt1 = (int(x * image_scale), int(y * image_scale))
                pt2 = (int((x + w) * image_scale), int((y + h) * image_scale))
                cv.Rectangle(ipl_img, pt1, pt2, cv.RGB(0, 0, 255), 3, 8, 0)
                #face_region = cv.GetSubRect(ipl_img,(x,int(y + (h/4)),w,int(h/2)))

            cv.SetImageROI(ipl_img, (pt1[0],
                    pt1[1],
                    pt2[0] - pt1[0],
                    int((pt2[1] - pt1[1]) * 0.7)))

            eyes = cv.HaarDetectObjects(ipl_img, eyeCascade,
            cv.CreateMemStorage(0),
            haar_scale, min_neighbors,
            haar_flags, (15,15))

            if eyes:
                # For each eye found
                for eye in eyes:

                    # Draw a rectangle around the eye
                    cv.Rectangle(ipl_img,                                       #image
                    (eye[0][0],                                                 #vertex pt1
                    eye[0][1]),
                    (eye[0][0] + eye[0][2],                                     #vertex pt2 opposite to pt1
                    eye[0][1] + eye[0][3]),
                    cv.RGB(255, 0, 0), 1, 4, 0)                                 #color,thickness,lineType(8,4,cv.CV_AA),shift


        cv.ResetImageROI(ipl_img)

        return ipl_img
##==============================================================================
    def setOnoff(self,switch):
        self.onoff = switch
        self.update_video()
##==============================================================================
    def setDetection(self):
        self.detect = not self.detect
##==============================================================================
    def update_video(self):
        self.image = None
        self.position.set("Face position x = " + self.getFaceX() + "    y = " + self.getFaceY())
        while self.onoff==True:
            if self.detect == False:
                (self.readsuccessful,self.f) = self.cam.read()                  #=>'numpy.ndarray' object
                self.capture = cv2.cvtColor(self.f, cv2.COLOR_BGR2RGBA)         #cv2.COLOR_RGB2GRAY)=>'numpy.ndarray' object

  
                self.draw_str(self.capture, (20, 50), 'Obraz testowy')#. FPS='+str(prop))

                self.a = Image.fromarray(self.capture)                          #new in PIL 1.1.6 (PIL.Image.VERSION=1.1.7)
                #print type(self.a) => <type 'instance'>
                self.b = ImageTk.PhotoImage(image=self.a)                       #type(self.b) => <type 'instance'>(of PhotoImage)
            if self.detect == True:                                             # face detection turned on
                haars = self.loadHaars()
                image = self.detectFace(self.cam, haars[0], haars[1], haars[2])
                tk_image = self.ipl2tk_image(image)
                self.image = self.canvas.create_image(0,0,image=tk_image,anchor=tk.NW)
                app.position.set("Face position x = " + self.getFaceX() + "    y = " + self.getFaceY())
            elif self.effect != 'none':                                         # effect button pressed
                self.b = self.applyEffect(self.f,int(self.cam.get(cv.CV_CAP_PROP_FRAME_WIDTH)),
                int(self.cam.get(cv.CV_CAP_PROP_FRAME_HEIGHT)))
                self.image = self.canvas.create_image(0,0,image=self.b,anchor=tk.NW)
            else:                                                               # face detection turned off
                img = self.drawHistogram(self.capture)
                self.histogram = self.histCanvas.create_image(0,0,image=img,anchor=tk.NW)
                self.image = self.canvas.create_image(0,0,image=self.b,anchor=tk.NW)

            self.root.update()
##==============================================================================
    def applyEffect(self, image, width, height):
        ipl_img = cv2.cv.CreateImageHeader((image.shape[1], image.shape[0]), cv.IPL_DEPTH_8U,3)
        cv2.cv.SetData(ipl_img, image.tostring(),image.dtype.itemsize * 3 * image.shape[1])

        gray = cv.CreateImage((width, height), 8, 1)                #tuple as the first arg

        dst_img = cv.CreateImage(cv.GetSize(ipl_img), cv.IPL_DEPTH_8U, 3)#_16S  => cv2.cv.iplimage
        if self.effect == 'dilate':
           cv.Dilate(ipl_img, dst_img,None,5)
        elif self.effect == 'laplace':
             cv.Laplace(ipl_img, dst_img,3)
        elif self.effect == 'smooth':
             cv.Smooth(ipl_img, dst_img, cv.CV_GAUSSIAN)
        elif self.effect == 'erode':
             cv.Erode(ipl_img, dst_img, None, 1)

        cv.Convert(dst_img,ipl_img)
        return self.ipl2tk_image(dst_img)
##==============================================================================
##>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
##==============================================================================
class App:
    def __init__(self, master,cam):
        self.cam = cam
        self.master = master
        self.frame = None
        self.canvas = None
        self.histCanvas = None
        self.video = None
        self.position = None



##window frame
        self.frame = tk.LabelFrame(self.master,text='Captured video')
        self.frame.pack()

##toolbar
        self.toolbar = tk.Frame(self.master)
        self.toolbar.configure(background='grey',borderwidth=2)
        self.toolbar.pack(side=tk.BOTTOM,padx=5,pady=5)                       #),expand=tk.YES,fill=tk.BOTH)



##adding buttons to toolbar
        self.button = tk.Button(self.toolbar, text="QUIT", fg="red", command=master.destroy)
        self.button.configure(background='tan')
        self.button.pack(side=tk.LEFT,padx=5, pady=5)
## ttk Combobox
        self.efLabel = Labelframe(self.toolbar, text="Choose an effect:")
        self.efLabel.pack(anchor = tk.W, padx=5, pady=2)
        self.efCombo = Combobox(self.efLabel, values = ['none','erode','smooth','dilate','laplace','threshold_otsu'], state='readonly')
        self.efCombo.current(0)
        self.efCombo.bind('<FocusIn>', self._update_values)
        self.efCombo.pack(anchor=tk.NW,padx=5, pady=5)

##fps

## for using of command binding see: 'Thinkink in Tkinter' tt077.py
        self.camon = tk.Button(self.toolbar, text="CAM on", fg="darkgreen", command=lambda: self.video.setOnoff(True))
        self.camon.configure(background='tan')
        self.camon.pack(side=tk.LEFT,padx=5, pady=5)

        self.camoff = tk.Button(self.toolbar, text="CAM off", fg="blue", command=lambda: self.video.setOnoff(False))
        self.camoff.configure(background='tan')
        self.camoff.pack(side=tk.LEFT,padx=5, pady=5)

        self.detector = tk.Button(self.toolbar, text="detect face", fg="blue", command=lambda: self.video.setDetection())
        self.detector.configure(background='tan')
        self.detector.pack(side=tk.LEFT,padx=5, pady=5)

        self.effect = tk.Button(self.toolbar, text="effect", fg="yellow", command=lambda: self.video.setEffect(self.efCombo.get()))
        self.effect.configure(background='tan')
        self.effect.pack(side=tk.LEFT,padx=5, pady=5)

        self.hi_there = tk.Button(self.toolbar, text="Hello")                   #, command=self.say_hi)
        self.hi_there.bind("<Control-Button-1>", self.say_hi)                   #event binding
        self.hi_there.configure(background='tan')
        self.hi_there.pack(side=tk.LEFT,padx=5, pady=5)
##canvas to draw on
        self.canvas = tk.Canvas(self.frame, width=620,height=460)
        self.canvas.configure(background="black",relief='ridge',highlightthickness=5,borderwidth=5)
        self.canvas.pack(side=tk.RIGHT,padx=5,pady=5)                             #(expand=tk.YES,fill=tk.BOTH)
##canvas to draw histogram
        self.histLabel = Labelframe(self.frame, text="Histogram")
        self.histLabel.pack(anchor = tk.W, padx=5, pady=2)
        self.histCanvas = tk.Canvas(self.histLabel, width=300,height=240)
        self.histCanvas.configure(background="black",relief='ridge',highlightthickness=5,borderwidth=5)
        self.histCanvas.pack(side=tk.TOP,padx=5,pady=5)
##sliders
        var=tk.DoubleVar()
        self.contrast = tk.Scale(self.frame, orient=tk.HORIZONTAL,
        label='Contrast',variable=var,resolution=0.5,from_=0.0, to=100.0, command=self._update_contrast)
        self.contrast.pack(side=tk.LEFT, anchor=tk.NW, padx=5, pady=5)
        self.brightness = tk.Scale(self.frame, orient=tk.HORIZONTAL,
        label='Brightness',from_=0, to=100, command=self._update_brightness)
        #self.brightness.bind('<FocusIn>', self._update_brightness)
        self.brightness.pack(side=tk.LEFT, anchor=tk.NW, padx=5, pady=5)
##position label
        self.position = tk.StringVar()
        self.xyLabel = tk.Label(self.toolbar, textvariable = self.position, fg='red',width=30,justify='left').pack(padx=1, pady=5)

##set the camera
        self.video = Camera(self.cam,self.master,self.canvas,self.histCanvas,self.frame,self.position)
        self.video.setOnoff(False)
##==============================================================================
##pooling video from camera
    def pool(self):
        if self.video != None:
            self.id=self.master.after(33,lambda: self.video.update_video())

##            self.master.after(50, lambda: self.pool())

##==============================================================================
##for test purposes only
    def say_hi(self, event):
        print "hi there, everyone!"
##==============================================================================
##combo event
    def _update_values(self, evt):
        # add entered text to combobox list of values
        widget = evt.widget           # get widget
        txt = widget.get()            # get current text
        #vals = widget.cget('values')  # get values

        print txt
        #print vals
##for editable widgets: update list of items with entered value
##        if not vals:
##            widget.configure(values = (txt, ))
##        elif txt not in vals:
##            widget.configure(values = vals + (txt, ))

        return 'break'  # don't propagate event
##==============================================================================
    def _update_brightness(self, val):
        self.video.setBrightness(float(val))
##==============================================================================
    def _update_contrast(self, val):
        self.video.setContrast(float(val))

##==============================================================================
##>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
##==============================================================================
if __name__ == '__main__':
    root = tk.Tk(className='Camera test by A Kryński')
    cam_capture = cv2.VideoCapture()
    cam_capture.open(1)

    app = App(root,cam_capture)
    app.pool()
    app.video.printCamResolution(cam_capture)


    root.mainloop()

    app.video = None
    app.cam = None
    del cam_capture
