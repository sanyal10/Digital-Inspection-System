#main script
from PIL import Image
from ultralytics import YOLO
import cv2
import numpy as np


# Load a pretrained YOLOv8n model
model = YOLO(r'C:\Users\DEBAJYOTI\OneDrive\Desktop\project_finalyear\Building facade.v2i.yolov8\runs-20231118T161130Z-001\runs\detect\train\weights\best.pt')


# Run inference on 'bus.jpg'
#results = model(r'C:\Users\DEBAJYOTI\OneDrive\Desktop\yolo\datasets\door.v1i.yolov8\valid\images\door33_jpg.rf.0a25321c5b8ec9e23fbb5886d6eafcac.jpg')  # results list

results=model.predict(r'C:\Users\DEBAJYOTI\OneDrive\Desktop\project_finalyear\Building facade.v2i.yolov8\train\images\156_A_jpg.rf.fd97ccff78ae969f9a9b7507b23a7208.jpg',save=False, imgsz=640, show_labels=True, conf=0.5,iou=0.75,show_conf=True)

print("classnames---->",results[0].names)
print("boxes_position---->",results[0].boxes.xyxy )
print("boxes_conf---->",results[0].boxes.conf)
print("boxes_classes---->",results[0].boxes.cls)
print("boxes_dimension---->",results[0].boxes.xywh)

#boxes in xywh format
img_box=results[0].boxes.xywh
img_box=img_box.tolist()
print('image boxes \n',img_box) #image boxes


# Show the results
for r in results:
    original_image = r.plot()  # plot a BGR numpy array of predictions
    im = Image.fromarray(original_image[..., ::-1])  # RGB PIL image
    im.show()  # show image
    #im.save('results.jpg')  # save image



######GUI for pixels measurement 

# define a null callback function for Trackbar
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



#original_image = cv2.imread(r'C:\Users\DEBAJYOTI\OneDrive\Desktop\object det\runs\detect\predict\240_A_jpg.rf.5f4981d12955b58c264b92c23c83e2d3.jpg')
clone = original_image.copy()

cv2.namedWindow('image')
cv2.setMouseCallback('image',extract_coordinates)
image_coordinates=[]

cv2.createTrackbar("user_input", "image", 0,500, null)
while(1):
    cv2.imshow('image',clone)
    user_value =cv2.getTrackbarPos("user_input", "image")
    if cv2.waitKey(20) & 0xFF == 27:
        break
print("user input value",user_value)
cv2.destroyAllWindows()

print("Data type of image-coordinates",type(image_coordinates[0]))

x0=image_coordinates[0][0]
y0=image_coordinates[0][1]
x1=image_coordinates[1][0]
y1=image_coordinates[1][1]

#print("x0 y0 x1 y1",x0,y0,x1,y1)

point1 = np.array((x0,y0))
point2 = np.array((x1,y1))

dist = np.linalg.norm(point1 - point2) #pixel distances

ft_pix=user_value/dist
 
# printing Euclidean distance
print('pixels distance',dist)  #pixel distance printing

print("length",img_box[0][3]*ft_pix)
print("breadth",img_box[0][2]*ft_pix)

image_coordinates.clear()

dimns=[]

for i in range(len(img_box)):
     x0=round(img_box[i][3]*ft_pix,2)
     y0=round(img_box[i][2]*ft_pix,2)
     dimns.append([x0,y0])

print('dimensions of window \n',dimns) #printing dimensions of every windoww 

#finding distance in-between images

#original_image = cv2.imread(r'C:\Users\DEBAJYOTI\OneDrive\Desktop\object det\runs\detect\predict\240_A_jpg.rf.5f4981d12955b58c264b92c23c83e2d3.jpg')
clone = original_image.copy()

cv2.namedWindow('image')
cv2.setMouseCallback('image',extract_coordinates)
image_coordinates=[]


while(1):
    cv2.imshow('image',clone)
    
    if cv2.waitKey(20) & 0xFF == 27:
        break


#print("image coordinates",image_coordinates)
cv2.destroyAllWindows()

x0=image_coordinates[0][0]
y0=image_coordinates[0][1]
x1=image_coordinates[1][0]
y1=image_coordinates[1][1]

#print("x0 y0 x1 y1",x0,y0,x1,y1)

point1 = np.array((x0,y0))
point2 = np.array((x1,y1))

dist = np.linalg.norm(point1 - point2)
 
# printing Euclidean distance
print("distance in real units",dist*ft_pix)