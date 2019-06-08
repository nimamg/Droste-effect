import cv2
import numpy as np

img = cv2.imread('clock.jpg', 1)
# img = np.zeros([10,10,3])
cv2.imshow('im',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
# print(img)
rows = np.linspace(start = -1, stop = 1, num = len(img))
cols = np.linspace(start = -1, stop = 1, num = len(img[0]))
x, y = np.meshgrid(cols, rows)
z = x + 1j*y
# print(z)
r1 = 0.2
r2 = 0.9
shape = z.shape
fz = z.flatten()
conds = [r1 <= abs(fz)]
chlist = [fz]
fz = np.select(conds,chlist)
conds = [r2 >= abs(fz)]
chlist = [fz]
fz = np.select(conds,chlist)
# print(fz)
# fz /= r1
# print(fz)
conds = [abs(fz) == 0,abs(fz) != 0 ]
chlist = [1,fz]
fz = np.select(conds,chlist)
# print(fz)
# np.clip(fz,1,r2/r1,fz)
fw = np.log(fz)
# print(fw)
w = fw.reshape(shape)
wx = np.real(w)
wy = np.imag(w)
wxMax = np.max(abs(wx))
wyMax = np.max(abs(wy))
xNew = (wx / wxMax + 1) * len(img[0]) / 2
yNew = (wy / wyMax + 1) * len(img) / 2
np.clip(xNew,0,len(img) - 1,xNew)
np.clip(yNew,0,len(img[0]) - 1,yNew)
np.floor(xNew,xNew)
np.floor(yNew,yNew)
lgMapped = np.empty(img.shape,dtype='uint8')
print(lgMapped.shape)
xNew = xNew.astype(int)
yNew = yNew.astype(int)
for i in range(len(xNew)):
    for j in range(len(yNew)):
        lgMapped[yNew[i][j]][xNew[i][j]] = img[i][j]
# print(img)
cv2.imshow('im',lgMapped)
cv2.waitKey(0)
cv2.destroyAllWindows()
