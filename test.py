import librosa
import numpy as np
import matplotlib
import librosa.display
import matplotlib.pyplot as plt

import pylab
matplotlib.use('Agg')
y, sr = librosa.load("happy.wav")
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
print("sample rate ", sr)

print("spectogram shape ", S.shape)
plt.figure(figsize=(10, 4))
librosa.display.specshow(librosa.power_to_db(S, ref=np.max), y_axis='mel', fmax=8000, x_axis='time')
# plt.colorbar(format='%+2.0f db')
# plt.title('Mel spectogram')
# plt.tight_layout()
# plt.show()
# plt.savefig("happy.jpg")
pylab.axis("off")
pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])
pylab.savefig("happy.jpg", bbox_inches=None, pad_inches=0)
pylab.close()
import cv2
img = cv2.imread("happy.jpg", 1)
print(img.shape)
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()