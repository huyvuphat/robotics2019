import cv2
import matplotlib.pyplot as plt
cap = cv2.VideoCapture(0)


ret, frame = cap.read()
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
# plt.imshow(rgb)
# plt.show()

print(rgb)

cap.release()
cv2.destroyAllWindows()
