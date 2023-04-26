from pickle import OBJ
import tkinter
import cv2
import PIL.Image
import PIL.ImageTk
from datetime import datetime
import os
import numpy as np
from joblib import load
import matplotlib.pyplot as plt


def check_for_templates(path):
    templates_paths = []
    templates_names = []
    temp = os.listdir(path)
    for files in temp:
        index = files.rfind('.')
        names = files[:index]
        templates_names.append(names)
        files = path+ '\\'+files
        templates_paths.append(files)
    return templates_paths, templates_names

def apply_filter(image, filter):                                                    # general function to apply filter. To add filter, filtername should be added to filter_names variable
    kernel = np.ones((5,5), np.uint8)                                               # matrix for transformation errosion
    if filter == 'dilate':                                                          #check what is value of Checkbox (by method in certain object) and apply filter based on that
        image = cv2.dilate(image, kernel, iterations=1)                             # apply filter on image
    elif filter == 'erode':
        image = cv2.erode(image, kernel, iterations=1)
    elif filter == 'grayscale':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif filter == 'denoise':
        image = cv2.fastNlMeansDenoising(image,None,10,7,7)
    elif filter == 'blur':
        image = cv2.blur(image, (3,3))
    elif filter == 'gaussian_blur':
        image = cv2.GaussianBlur(image, (3,3), 0)
    elif filter == 'threshold':
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # th_value = tkinter.IntVar()
        # threshold_value = tkinter.Scale(App, orient='horizontal', variable=th_value, activebackground = 'darkgray', highlightbackground='darkgray') #zmienna przetrzymujaca wartosc z suwaka to ''self.threshold''
        # threshold_value.grid(column=600, row=600)
        # threshold_value.set(10)        
        th, image = cv2.threshold(image, 180, 180, cv2.THRESH_TRUNC)
        
    elif filter == '0':
        pass
    return image




filters_names = ['dilate', 'erode', 'grayscale', 'denoise', 'blur', 'gaussian_blur', 'threshold']
templates_instances = []
scale_instances = []
checkbox_instances = []
template_names_path = 'D:\\Aplikacje PYTHON\Android_camera\\templates'
templates_paths = []
templates_names = []
templates_path, templates_names = check_for_templates(template_names_path)
detector = cv2.ORB_create(nfeatures=1000)
detector_sift  = cv2.SIFT_create()
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
threshold = 30
clf = load('D:\\Aplikacje PYTHON\\Android_camera\\PRZEDMIOTY.joblib')

class App:
    #def __init__(self, window, window_title, video_source='http://192.168.41.43:8080/video'):
    def __init__(self, window, window_title, video_source=1):
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source
        self.vid = VideoCapture(self.video_source)  #call videosource class, create instance with desired video_source
        self.canvas = tkinter.Canvas(window, width= self.vid.width, height= self.vid.height)    #create canvas equas to video size
        self.canvas.grid( row=1, column=1, rowspan=480, columnspan=640)
        self.filter_label = tkinter.Label(text='Apply filter', width=20, height=4, borderwidth=2, relief='solid')
        self.filter_label.grid(column=1, row=int(self.vid.height+1), rowspan=int(len(filters_names)))

        self.width= str(int(self.vid.width+200))
        self.height= str(int(self.vid.height) + (int(len(filters_names)*30))) #must be height of video + height of #buttons * height of buttons 
        self.geometry = self.width +'x'+self.height
        self.window.geometry(self.geometry)                 #geometry of main window is video + 200 px
        
        self.row_right_panel = 1
        self.row_left_panel = int(self.vid.height)+1
        
        for instance in templates_instances:                                          #initialize scale controls
            self.column = int(self.vid.width)+1
            scale_instance = 'scale'+instance.name
            Scale(window,scale_instance, self.column, self.row_right_panel)
            self.row_right_panel += 20
        print('initialized scales ', scale_instances)
        
        checkbox_instance = ''
        for filter in filters_names:                                        #initialize filters_checkbox
            checkbox_instance = 'Checkbox_'+str(filter)
            print(checkbox_instance)
            Checkbox(window, checkbox_instance, filter, 22, self.row_left_panel)
            self.row_left_panel += 1

        Snap_Button = tkinter.Button(window, command= lambda:self.take_snap(), width=10, text='Take Snap!')
        Snap_Button.grid(column=1, row=self.row_left_panel)
        self.row_left_panel += 1
        ML_button = tkinter.Button(window, command= lambda:self.ML_classification(), width=10, text='Zweryfikuj z pomocą modelu ML!')
        ML_button.grid(column=1, row=self.row_left_panel)
        self.row_left_panel += 1

        self.delay = 30     #delay to refresh window with self.update() function
        self.update()
        self.window.mainloop()

    def update(self):                                   #recurrent function, update frame and calls itself after certain delay

        ret, frame = self.vid.get_frame()               #call get_frame method of vid instance of VideoCapture object        
        for checkbox in checkbox_instances:             #check checkboxes values and apply filter to videoframe
            frame = apply_filter(frame, checkbox.get_Checkbox_value())  #do filter by self.apply_filter function

        self.text_position = 30                         # position for text Match! on vieoframe
        self.text = ''                                  # empty text at initialization (before anything is matched)
        if ret:
            kp_frame, des_frame = detector.detectAndCompute(frame, None)                                    # get keypoints and descriptor for videoframe
            #frame = cv2.drawKeypoints(frame, kp_frame, None, color=(255,00,0), flags=0)                     # draw key points on frame, need to be after adding

            for template, scale_instance in zip(templates_instances, scale_instances):      # go through templates & scales with thresholds to check for matches
                self.des1 = template.des                                                    # get descriptor from template
                self.kp1 = template.kp                                                      # key points of template
                name_of_object = template.name                                              # get name of object to be detected == name of photo in template folder
                scale_instance.get_threshold()                                              # get threshold value from slider(scale)
                template.threshold = scale_instance.get_threshold()                         # set object attirbute threshold same as picked from silder(scale)

                if type(des_frame) != None:                                                                 # if descriptor is NOT empty try matching

                    #################ORB DETECTOR MATCHING#################
                    try:
                        matches = matcher.match(self.des1,des_frame)                                                  # find matches in between descriptor of template and frame
                        matches = sorted(matches, key = lambda x:x.distance)                                          # sort matches low (best) to high (bad)                        
                        if matches[0].distance < template.threshold:                                                  # if lowest matcher 'distance' is lower than threshold == we have match!
                            score = matches[0].distance
                            score=str(score)
                            self.text = 'Match! '+'    '+ name_of_object+'    '+score                                 # update text to view 'Match 'object' 'score' '  
                            ##### rysowanie ramki na odnalezionym obiekcie #####
                            src_pts  = np.float32([self.kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
                            dst_pts  = np.float32([kp_frame[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
                            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
                            h,w = template.h, template.w
                            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
                            dst = cv2.perspectiveTransform(pts,M)
                            frame = cv2.polylines(frame, [np.int32(dst)], True, (150,255,150), 1, cv2.LINE_AA)
                            frame = cv2.putText(img = frame, org=(int(dst[0][0][0]),int(dst[0][0][1])), text = self.text, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.8, color=(150,255,150), thickness=2)
                            ##### rysowanie ramki na odnalezionym obiekcie #####

      
                    #################ORB DETECTOR MATCHING#################
                    except Exception as e:
                        print(e)
                        pass

                #cv2.putText(img = frame, org=(30,self.text_position), text=self.text, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.8, color=(0,0,200), thickness=2)       

                self.text_position += 30
                self.text = ''
            
            self.photo = PIL.ImageTk.PhotoImage(image= PIL.Image.fromarray(frame))                  # after matching, putting text etc, long transformation to create frame in canvas
            self.canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)
        self.window.after(self.delay, self.update)                                                  #call this function again with delay as set

    def take_snap(self):
        print('button klikniety')
        ret, frame = self.vid.get_frame()               #call get_frame method of vid instance of VideoCapture object       
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #for checkbox in checkbox_instances:             #check checkboxes values and apply filter to videoframe
        #    frame = apply_filter(frame, checkbox.get_Checkbox_value())  #do filter by self.apply_filter function
        now = datetime.now()
        name = now.strftime('%H_%M_%S')
        name = str(name)
        print(name)
        cv2.imshow(name, frame)
        cv2.imwrite(f'templates\\{name}.jpg', frame)
        pass

    def ML_classification(self):
        ret, frame = self.vid.get_frame()               #call get_frame method of vid instance of VideoCapture object  
        frame  = cv2.fastNlMeansDenoising(frame,None,10,7,7)     
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frameh = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame2 = frame.reshape(1,-1)
        plt.clf()
        color = ('b','g','r')
        for i,col in enumerate(color):
            histr = cv2.calcHist([frameh],[i],None,[256],[0,256])
            np.append(frame2, histr)
        plt.plot(histr,color = col)
        plt.show()
        y_pred = clf.predict(frame2)
        y_pred = str(y_pred)
        cv2.putText(img=frame, org=(150,150), text = y_pred, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(255, 255, 0), thickness=2)
        cv2.imshow('predykcja', frame)
        print(y_pred, '     ')
        pass
    
class Checkbox:
    def __init__(self, window, checkbox_instance, filter, column, row_left_panel):
        self.name = str(checkbox_instance)
        checkbox_instances.append(self)
        self.variable = tkinter.StringVar()
        self.checkbox_instance = tkinter.Checkbutton(window, text=filter, variable=self.variable, onvalue=filter, command=self.apply_filter, offvalue='_', justify='left')
        self.checkbox_instance.deselect()
        self.checkbox_instance.grid(column=column, row=row_left_panel)
    def get_Checkbox_value(self):                                               #needto have that function to get variable out of this object 
        return self.variable.get()
    def apply_filter(self):                         # to optimize filter need to be applied 'on click', not on every refresh (it is laggy this way)
        self.filters = ''
        print(self.filters)
        for checkbox in checkbox_instances:
            if checkbox.get_Checkbox_value() != 0:
                self.filters = self.filters + ' ' + checkbox.get_Checkbox_value()
        print(self.filters)
        for template in templates_instances:
            template.filter(self.filters)
##### end of class


class Scale:
    def __init__(self, window, scale_instance, column, row_right_panel):
            scale_instances.append(self)
            self.name = scale_instance
            self.variable = tkinter.IntVar()
            self.label = tkinter.Label(text = 'Threshold for object: ' + self.name )
            self.label.grid(column=column, row=row_right_panel)
            self.scale_instance = tkinter.Scale(window, orient='horizontal', variable=self.variable, activebackground = 'darkgray', highlightbackground='darkgray') #zmienna przetrzymujaca wartosc z suwaka to ''self.threshold''
            self.scale_instance.grid(column=column, row=row_right_panel+10)
            self.scale_instance.set(10)

    def get_threshold(self):
        return self.variable.get()
##### end of class

class VideoCapture:
    def __init__(self, video_source):
        self.vid = cv2.VideoCapture(video_source)   #get video from source 
        if not self.vid.isOpened():
            raise ValueError('Unable to open video source ', video_source)

        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)     #get width of video
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)   #get height of video
    
    def get_frame(self):                                        #get frame from video
        if self.vid.isOpened():                                 
            ret,frame = self.vid.read()                         #get flag and frame
            if ret:
                return(ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) # ret is boolean flag, second position is frame in RGB
            else:
                return (ret, None)                              #if video is opened but ret flag is false
        else:
            return(ret, None)                                   # if video is NOT opened

    def __del__(self):                                          #release camera when videocapture is close
        if self.vid.isOpened():
            self.vid.release()
##### end of class

class Template:    
    def __init__(self, name, image, filter):
        self.name = name
        self.path = image
        self.image = cv2.imread(self.path)
        templates_instances.append(self)
        self.kp, self.des = detector.detectAndCompute(self.image, None)
        self.image = cv2.drawKeypoints(self.image, self.kp, None, color = (0,200,0), flags=0)               # draw detected keypoints on template
        self.threshold = 10
        self.filter(filter)
        self.h, self.w = self.image.shape[:2]

    def filter(self,filter):
        self.image = cv2.imread(self.path)
        for checkbox in checkbox_instances:
            self.image = apply_filter(self.image, checkbox.get_Checkbox_value())           
        self.text = str(filter)
        if self.text == '0': self.text =''
        self.kp, self.des = detector.detectAndCompute(self.image, None)
        self.image = cv2.drawKeypoints(self.image, self.kp, None, color = (0, 150, 0), flags=0)               # draw detected keypoints on template
        cv2.putText(img = self.image, org=(50,50), text=self.text, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(0, 150, 0), thickness=3)
        cv2.imshow(self.name, self.image)
 ##### end of class
         
for path, name in zip(templates_path, templates_names):
    print('Inicjalizuje obiekt ', name, ' ze sciezki ', path)
    Template(name, path, '')

App(tkinter.Tk(), 'Live Video')
