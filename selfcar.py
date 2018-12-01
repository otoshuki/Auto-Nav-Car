#Kriti - Kameng 2018

#Import required libraries
import cv2
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import time
import serial
import struct

#Take input from camera
cap = cv2.VideoCapture('abc.mp4')
ret, frame = cap.read()
#frame = cv2.imread('Lane8.png')
font = cv2.FONT_HERSHEY_SIMPLEX

#Global Variables
size_y,size_x,channels = frame.shape

#Nothing function for trackbars
def nothing(x):
    pass

#arduino = serial.Serial('/dev/ttyACM0', 9600)

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#Detection and Planning
def run(thresh):
    global size_x
    global size_y
    global crop_y
    start = time.time()
    print('Starting Detection')
    time.sleep(1)
    #Run continuously
    while True:
        #Get image
        ret, frame = cap.read()
        crop_top = thresh[0]
        crop_bottom = thresh[1]
        lower = thresh[2]
        upper = thresh[3]
        #Crop Image
        crop = frame[size_y-crop_top:size_y-crop_bottom,0:size_x]
        crop_y,crop_x,crop_ch = crop.shape
        centre_x = int(size_x/2)
        centre_y = int(size_y/2)
        #Convert to grayscale
        gray = cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
        #Create binary mask
        mask = cv2.inRange(gray,lower,upper)
        #Morphological transformations
        mask = morph(3,mask)
        #Lists to store left and right values
        left = []
        right = []
        #Sample at specific points along y axis
        for i in range(crop_y):
            if i%5 == 0:
                #Get last black point left to centre
                l_max = centre_x - mask[i,0:centre_x][::-1].argmax()
                left.append([l_max,i + size_y-crop_y])
                #Get first black point right to centre
                r_max = mask[i,centre_x:crop_x].argmax()
                right.append([centre_x + r_max,i + size_y-crop_y])
        left = np.array(left)
        right = np.array(right)
        #print(right[int(len(right)/2),0], size_x)
        #Check for 90 degrees to left
        if left[0,0] == centre_x:
            if right[0,0] == centre_x:
                transmit(-90)
                continue
        #Regression
        l_coeff = regression(left)
        r_coeff = regression(right)
        #Find intersection
        intercept_x = (r_coeff[1] - l_coeff[1])/(l_coeff[0] - r_coeff[0])
        intercept_y = l_coeff[0]*intercept_x + l_coeff[1]
        #Find angle wrt centre
        angle = int(np.arctan2((centre_x-intercept_x),(-1*intercept_y + size_y))[0] * 180/3.14)
        #print('Angle : ', angle)
        #Draw centre and intersection lines
        new = frame.copy()
        l_x1 = (size_y-l_coeff[1])/l_coeff[0]
        l_y1 = size_y
        l_x2 = -l_coeff[1]/l_coeff[0]
        r_x1 = (size_y-r_coeff[1])/r_coeff[0]
        r_y1 = size_y
        r_x2 = -r_coeff[1]/r_coeff[0]
        cv2.line(new,(l_x1,l_y1),(l_x2,0),(0,0,255),2)
        cv2.line(new,(r_x1,r_y1),(r_x2,0),(0,0,255),2)
        cv2.line(new,(centre_x,size_y),(centre_x,0),(255,255,255),2)
        cv2.line(new,(centre_x,size_y),(intercept_x,intercept_y),(255,255,0),2)
        cv2.putText(new,str(angle),(centre_x,size_y-5),font,0.5,(0,255,0),1,cv2.LINE_AA)
        #Show images
        cv2.imshow('NEW',new)
        #Transmit Speeds
        transmit(angle)
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
    #regression(left)
    cap.release()
    cv2.destroyAllWindows()

#Kernel function
def kernel(size):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(size,size))

#Transformations
def morph(layers,mask):
    #Max 4
    if layers == 1:
        mask = cv2.GaussianBlur(mask,(5,5),2)
    if layers == 2:
        mask = cv2.GaussianBlur(mask,(5,5),2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel(4))
    if layers == 3:
        mask = cv2.GaussianBlur(mask,(5,5),2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel(4))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel(6))
    if layers == 4:
        mask = cv2.GaussianBlur(mask,(5,5),2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel(4))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel(6))
        mask = cv2.medianBlur(mask, 5)
    return mask

#Regression
def regression(data):
    X_train = data[:,0].reshape(-1,1)
    y_train = data[:,1]
    regr = linear_model.LinearRegression()
    regr.fit(X_train,y_train)
    #print('Regression Coefficents : ', regr.coef_)
    #print('Intercept : ', regr.intercept_)
    m = regr.coef_
    c = regr.intercept_
    if m == 0:
        m += 0.0001
    return [m,c]

#Calibration
def calibrate():
    print('Starting Calibration')
    ret,frame = cap.read()
    while True:

        #Get trackbar values
        crop_top = int(cv2.getTrackbarPos('Crop_top', 'Thresholds'))
        crop_bottom = int(cv2.getTrackbarPos('Crop_bottom', 'Thresholds'))
        lower = int(cv2.getTrackbarPos('Lower_Gray', 'Thresholds'))
        upper = int(cv2.getTrackbarPos('Upper_Gray', 'Thresholds'))
        crop = frame[size_y-crop_top:size_y-crop_bottom,0:size_x]
        crop_y,crop_x,crop_ch = crop.shape
        centre_x = int(size_x/2)
        #Convert to grayscale
        gray = cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
        #Create binary mask
        mask = cv2.inRange(gray,lower,upper)
        #Morphological transformations
        mask = morph(3,mask)
        cv2.imshow('Mask',mask)
        cv2.imshow('Frame',gray)
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
    #cap.release()
    cv2.destroyAllWindows()
    out = [crop_top,crop_bottom,lower,upper]
    print('Calibration done')
    print('Selected Parameters')
    print('crop_top :',crop_top, '\tcrop_bottom : ',crop_bottom)
    print('lower :',lower,'\tupper :',upper)
    return out

#Transmission
def transmit(angle):
    #Speed control
    if (abs(angle)<5):
        speed_l = 2
        speed_r = 2
        dir_l = 0 #Forward
        dir_r = 0 #Forward
        print('Motion : Forward')
    if (abs(angle) >= 90):
        speed_l = 2
        speed_r = 2
        print('Motion : 90 Degrees')
    #Set for positive angle - Left motion
    if angle > 5 :
        dir_l = 1 #Backward
        dir_r = 0 #Forward
        print('Motion : Left')
        if angle < 60:
            speed_l = 1
            speed_r = 1
    #In case 60<angle<90 both have same speeds but opppsite directions
    #Set for negative angle - Right motion
    elif angle < -5 :
        print('Motion : Right')
        dir_l = 0 #Forward
        dir_r = 1 #Backward
        if angle > -60:
            speed_l = 1
            speed_r = 1
    #Send 4 digit number
    #serial_data = str(dir_l) + str(speed_l) + str(dir_r) + str(speed_r)
    #print(serial_data)
    #arduino.write(struct.pack('>BBBB',dir_l,speed_l,dir_r,speed_r))

#Main function
def main():
    print('Kameng Hostel - Autonomous Navigation System\n')
    #Create window for thresholds
    cv2.namedWindow('Thresholds')
    cv2.createTrackbar('Crop_top', 'Thresholds', size_y, size_y, nothing)
    cv2.createTrackbar('Crop_bottom', 'Thresholds', 0, size_y, nothing)
    cv2.createTrackbar('Lower_Gray', 'Thresholds', 0, 255, nothing)
    cv2.createTrackbar('Upper_Gray', 'Thresholds', 255, 255, nothing)
    thresh = calibrate()
    time.sleep(1)
    run(thresh)
    print('Runtime Over!')
    print('Closing!')

#Run the program
if __name__ == '__main__':
    main()
