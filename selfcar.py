"""
1. Build Car Model
2. Find the track boundaries
?. Move the car
"""
#Import required libraries
import cv2
import numpy as np

#Take input from camera
#cap = cv2.VideoCapture(0)

#Lane detection
def detect():
    print('not done yet')
    while True:
        #Take camera input
        #ret,frame = cap.read()
        frame = cv2.imread('lane3.png')
        #Convert to HSV format
        hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        #Find the tracklines
        #black_lower = np.array([0,0,0])
        #black_upper = np.array([180,255,50])
        lower = np.array([0,0,0])
        upper = np.array([180,255,50])
        #Convert to binary image
        #bin = cv2.inRange(hsv,black_lower,black_upper)
        bin = cv2.inRange(hsv,lower,upper)
        #Apply morphological transformations
        bin = cv2.GaussianBlur(bin,(5,5),2)
        #Bitwise AND on Grayscale
        masked = cv2.bitwise_and(gray,gray,mask=bin)
        #Canny Edge detection
        canny_edges = cv2.Canny(bin,50,100)
        #Apply Hough Line Transform to get the tracklines
        hough = cv2.HoughLinesP(canny_edges, 1, np.pi/180, 100, np.array([]),100,200)
        #Get only the required lines
        #Draw the lines on feed
        show = frame.copy()
        for line in hough:
            for x1,x2,y1,y2 in line:
                cv2.line(show,(x1,y1),(x2,y2),(0,0,255),2)
        #Control over removed tracklines
        #Show the feed
        cv2.imshow('EDGES',canny_edges)
        cv2.imshow('EE', masked)
        cv2.imshow('FRAME',bin)
        cv2.imshow('GRAY',show)
        #waitKey
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
    #cap.release()
    cv2.destroyAllWindows()

#Motion Control Algorithm
def move():
    print('not done yet')
    #Get the lines from detection
    #Find the extended point of intersection
    #Angle wrt the centre gives the direction to move
    #Transmission


#Transmission
def trans():
    print('not done yet')
    #Start Serial
    #Serial output (x,y)

#Main function
def main():
    print('Not completed yet')
    detect()


#Run the program
if __name__ == '__main__':
    main()
