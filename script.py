from ultralytics import YOLO
import cv2

model = YOLO("C:/Users/Asus/Documents/my-data/combined/runs/detect/train4/weights/best.pt")

results = model("C:/Users/Asus/Documents/pt-projects/test.jpg")

img = results[0].plot()

cv2.imshow("Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
