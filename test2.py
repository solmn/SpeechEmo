import matplotlib.pyplot as plt
import cv2
import numpy as np

img = cv2.imread("happy.jpg")
print(img.shape)
pts1 = np.float32([[30,10],[980,10],[30,390],[980,390]])
pts2 = np.float32([[0,0],[1000,0],[0,400],[900,400]])

M = cv2.getPerspectiveTransform(pts1,pts2)

dst = cv2.warpPerspective(img,M,(1000,400))

plt.subplot(121),plt.imshow(img),plt.title('Input')
plt.subplot(122),plt.imshow(dst),plt.title('Output')
# plt.show()
# print(img.shape)
cv2.imshow('dst', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()