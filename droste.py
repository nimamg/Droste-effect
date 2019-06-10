import cv2
import numpy as np
import copy as cp

img = cv2.imread('clock.jpg', 1)
cv2.imshow('im',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

rows = np.linspace(start = -1, stop = 1, num = len(img[0]))
cols = np.linspace(start = -1, stop = 1, num = len(img))
x, y = np.meshgrid(cols, rows)
z = x + 1j*y

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
conds = [abs(fz) == 0,abs(fz) != 0 ]
chlist = [1,fz/r1]
fz = np.select(conds,chlist)
fw = np.log(fz)
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
xNew = xNew.astype(int)
yNew = yNew.astype(int)

for i in range(len(xNew)):
    for j in range(len(yNew)):
        lgMapped[yNew[i][j]][xNew[i][j]] = img[i][j]
cv2.imshow('im',lgMapped)
cv2.waitKey(0)
cv2.destroyAllWindows()

# tiledPic = np.tile(lgMapped, (3,1,1))
xNew = np.tile(xNew, (3,1))
yAdd = cp.deepcopy(yNew)
yAdd2 = cp.deepcopy(yNew)
yAdd += 400
yAdd2 += 800
yNew = np.concatenate((yNew, yAdd))
yNew = np.concatenate((yNew, yAdd2))
z = xNew + 1j * yNew
# cv2.imshow('im',tiledPic)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# print(tiledPic.shape)
# rows = np.linspace(start = -1, stop = 1, num = len(tiledPic))
# cols = np.linspace(start = -1, stop = 1, num = len(tiledPic[0]))
# x, y = np.meshgrid(cols, rows)
# z = x + 1j*y
alpha = np.arctan(np.log(r2/r1) / (2 * np.pi))
print(alpha)
f = np.cos(alpha)
rz = z * f * np.exp(1j*alpha)
rz = np.exp(rz)
rotX = np.real(rz)
rotY = np.imag(rz)
rxMax = np.max(abs(rotX))
ryMax = np.max(abs(rotY))
xNew2 = (rotX / rxMax + 1) * 400 / 2
yNew2 = (rotY / ryMax + 1) * 1200 / 2
np.floor(xNew2,xNew2)
np.floor(yNew2,yNew2)
rotatedPic = np.zeros((1200,400,3),dtype='uint8')
xNew2 = xNew2.astype(int)
yNew2 = yNew2.astype(int)
# print(xNew2, len(tiledPic), len(tiledPic[0]), rotatedPic.shape, tiledPic.shape)
np.clip(xNew2,0,400 - 1,xNew2)
np.clip(yNew2,0,1200 - 1,yNew2)
# print("HHHHHHHHHHHHH")
# print(len(xNew2), len(yNew2))
for p in range(1200):
    for q in range(400):
        if yNew[p][q] > 1200:
            print("Y")
        elif xNew[p][q] > 1200:
            print("x")
        rotatedPic[yNew2[p][q]][xNew2[p][q]] = img[p%400][q]

cv2.imshow('im',rotatedPic)
cv2.waitKey(0)
cv2.destroyAllWindows()
