#calibration of pixels units to real world dimensions


import cv2
import numpy as np
import streamlit as st


# define a null callback function for Trackbar
st.title("OpenCV Demo App")
st.subheader("This app allows you to play with Image filters!")
st.text("We use OpenCV and Streamlit for this demo")


def null(x):
    pass
def extract_coordinates(event, x, y,flags,parameters):
        global image_coordinates
        global clone
        global original_image
        # Record starting (x,y) coordinates on left mouse button click
        if event == cv2.EVENT_LBUTTONDOWN:
            image_coordinates = [(x,y)]
            print(image_coordinates)
        # Record ending (x,y) coordintes on left mouse bottom release
        elif event == cv2.EVENT_LBUTTONUP:
            image_coordinates.append((x,y))
            print('Starting: {}, Ending: {}'.format(image_coordinates[0], image_coordinates[1]))
            # Draw line
            cv2.line(clone,image_coordinates[0],image_coordinates[1], (36,255,12), 2)
            cv2.imshow("image",clone) 
        # Clear drawing boxes on right mouse button click
        elif event == cv2.EVENT_RBUTTONDOWN:
            clone = original_image.copy()

original_image = cv2.imread(r'C:\Users\DEBAJYOTI\OneDrive\Desktop\object det\runs\detect\predict\240_A_jpg.rf.5f4981d12955b58c264b92c23c83e2d3.jpg')
clone = original_image.copy()

#st.image(np.array(original_image))

#extracting image coordinates 
cv2.namedWindow('image')
cv2.setMouseCallback('image',extract_coordinates)
image_coordinates=[]

cv2.createTrackbar("user_input", "image", 0,500, null)
cv2.putText(clone, '1',(459,551),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA) 

while(1):
    cv2.imshow('image',clone)
    user_value =cv2.getTrackbarPos("user_input", "image")
    if cv2.waitKey(20) & 0xFF == 27:
        break
print(user_value)
cv2.destroyAllWindows()

print(type(image_coordinates[0]))

x0=image_coordinates[0][0]
y0=image_coordinates[0][1]
x1=image_coordinates[1][0]
y1=image_coordinates[1][1]

print("x0 y0 x1 y1",x0,y0,x1,y1)

point1 = np.array((x0,y0))
point2 = np.array((x1,y1))

dist = np.linalg.norm(point1 - point2)
 
# printing Euclidean distance
print(dist)
 
#####
original_image = cv2.imread(r'C:\Users\DEBAJYOTI\OneDrive\Desktop\object det\runs\detect\predict\240_A_jpg.rf.5f4981d12955b58c264b92c23c83e2d3.jpg')
clone = original_image.copy()

cv2.namedWindow('image')
cv2.setMouseCallback('image',extract_coordinates)
image_coordinates=[]


while(1):
    cv2.imshow('image',clone)
    
    if cv2.waitKey(20) & 0xFF == 27:
        break


    
print(user_value)
cv2.destroyAllWindows()

x0=image_coordinates[0][0]
y0=image_coordinates[0][1]
x1=image_coordinates[1][0]
y1=image_coordinates[1][1]

print("x0 y0 x1 y1",x0,y0,x1,y1)

point1 = np.array((x0,y0))
point2 = np.array((x1,y1))

dist = np.linalg.norm(point1 - point2)
 
# printing Euclidean distance
print(dist)
