import cv2
import numpy as np
import streamlit as st




def draw(event,x,y,flags,param):
    global coord1
    
    if event == cv2.EVENT_LBUTTONDOWN:
        coord1=(x,y)
        print(coord1)
    elif (event == cv2.EVENT_MOUSEMOVE and coord1!=()):
        #cv2.circle(img,(x,y),10,(255,0,0),-1)
        cv2.line(img,coord1,(x,y),(255,0,0),2)


img = np.zeros((512,512,3), np.uint8)
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw)
coord1=()

while(True):

    cv2.imshow('image', img)
    if cv2.waitKey(20) & 0xff == 27:
        break

cv2.destroyAllWindows()



r"""
img = np.zeros((512,512,3), np.uint8)
cv2.imshow('image',img)
points = []
def click_event(event, x, y, flags, params):
   if event == cv2.EVENT_LBUTTONDOWN:
      points.append((x,y))
      cv2.circle(img,(x,y), 1, (36,255,12), -1)
      if len(points) >= 2:
         cv2.line(img, points[-1], points[-2], (36,0,255), 2)
         cv2.imshow('image', img)
cv2.setMouseCallback('image', click_event)
while(True):
    cv2.imshow('image', img)
    if cv2.waitKey(20) & 0xff == 27:
        break
cv2.destroyAllWindows()
print(results[0].boxes.xyxy )
print(results[0].boxes.conf)
print(results[0].boxes.cls)
print(results[0].boxes.xywh)
c_x=int(img_box[0][0])
c_y=int(img_box[0][1])

print("cx",c_x,c_y)

original_image = cv2.imread(C:\Users\DEBAJYOTI\OneDrive\Desktop\object det\runs\detect\predict\240_A_jpg.rf.5f4981d12955b58c264b92c23c83e2d3.jpg')
cv2.namedWindow('image')
cv2.putText(original_image, '1',(c_x,c_y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA) 
while(1):
cv2.imshow('image',original_image)
if cv2.waitKey(20) & 0xFF == 27:
break






r"""
