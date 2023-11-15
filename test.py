import cv2
import numpy as np
def draw_circle(event,x,y,flags,param):
    global mouseX,mouseY
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(img,(x,y),100,(255,0,0),-1)
        mouseX,mouseY = x,y

img = np.zeros((512,512,3), np.uint8)
cv2.namedWindow(winname='my_drawing')

#doesn't understand the below code how can we call the draw_circle without passing parameters
cv2.setMouseCallback('my_drawing',draw_circle)

while(1):
    cv2.imshow('image',img)
    k = cv2.waitKey(20) & 0xFF
    if k == 27:
        break
    elif k == ord('a'):
        print(mouseX,mouseY)
        
cv2.destroyAllWindows()