import cv2
import math
import numpy as np

color_explore = np.zeros((150,150,3), np.uint8)  
color_selected = np.zeros((150,150,3), np.uint8)


#save selected color RGB in file
def thresh(R,G,B):
        thresh_val = math.ceil((0.2126*R + 0.7152*G + 0.0722*B))
        print(thresh_val)
        ret, thresh_img = cv2.threshold(img, thresh_val, 255, cv2.THRESH_TOZERO_INV)
        cv2.imshow('Thresholded Image', thresh_img)
        cv2.waitKey(1)

#Mouse Callback function
def show_color(event,x,y,flags,param): 
        
        B=img[y,x][0]
        G=img[y,x][1]
        R=img[y,x][2]
        color_explore [:] = (B,G,R)

        if event == cv2.EVENT_LBUTTONDOWN:
                color_selected [:] = (B,G,R)
                print(f"Selected: R={R}, G={G}, B={B}")


        if event == cv2.EVENT_RBUTTONDOWN:
                B=color_selected[10,10][0]
                G=color_selected[10,10][1]
                R=color_selected[10,10][2]
                print(R,G,B)
                thresh(R,G,B)
        

#live update color with cursor
cv2.namedWindow('color_explore')
cv2.resizeWindow("color_explore", 50,50);

#Show selected color when left mouse button pressed
cv2.namedWindow('color_selected')
cv2.resizeWindow("color_selected", 50,50);

#image window for sample image
cv2.namedWindow('image')

#sample image path
img_path="./My_ISL/A_Wall_1.jpg"

#read sample image
img_gray=cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
img=cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

#mouse call back function declaration
cv2.setMouseCallback('image',show_color)

#while loop to live update
while (1):
        
        cv2.imshow('image',img)
        cv2.imshow('gray image', img_gray)
        cv2.imshow('color_explore',color_explore)
        cv2.imshow('color_selected',color_selected)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
                break

cv2.destroyAllWindows()
