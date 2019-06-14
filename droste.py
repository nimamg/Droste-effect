import cv2
import numpy as np
import copy as cp

img = cv2.imread('clock.jpg', 1)
cv2.imshow('im',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Part 1
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
lgxNew = (wx / wxMax + 1) * len(img[0]) / 2
lgyNew = (wy / wyMax + 1) * len(img) / 2
np.clip(lgxNew, 0, len(img) - 1, lgxNew)
np.clip(lgyNew, 0, len(img[0]) - 1, lgyNew)
np.floor(lgxNew,lgxNew)
np.floor(lgyNew,lgyNew)
lgMapped = np.empty(img.shape,dtype='uint8')
lgxNew = lgxNew.astype(int)
lgyNew = lgyNew.astype(int)

for i in range(len(lgxNew)):
    for j in range(len(lgyNew)):
        lgMapped[lgyNew[i][j]][lgxNew[i][j]] = img[i][j]
cv2.imshow('im',lgMapped)
cv2.waitKey(0)
cv2.destroyAllWindows()
# End of Part 1

#Part 2
rows = np.linspace(start = -1, stop = 1, num = len(img[0]))
cols = np.linspace(start = -1, stop = 1, num = len(img))
x, y = np.meshgrid(cols, rows)
rotZ = x + 1j*y
rotZ = z * np.exp(1j * np.pi / 4)
rotX = np.real(rotZ)
rotY = np.imag(rotZ)
rotXmax = np.max(abs(rotX))
rotYmax = np.max(abs(rotY))
rotxNew = (rotX / rotXmax + 1) * len(img[0]) / 2
rotyNew = (rotY / rotYmax + 1) * len(img) / 2
np.floor(rotxNew,rotxNew)
np.floor(rotyNew,rotyNew)
rotxNew = rotxNew.astype(int)
rotyNew = rotyNew.astype(int)
rotatedPic = np.zeros(img.shape,dtype = 'uint8')
np.clip(rotxNew, 0, len(img) - 1, rotxNew)
np.clip(rotyNew, 0, len(img[0]) - 1, rotyNew)
for i in range(len(rotxNew)):
    for j in range(len(rotyNew)):
        rotatedPic[rotyNew[i][j]][rotxNew[i][j]] = img[i][j]
cv2.imshow('im',rotatedPic)
cv2.waitKey(0)
cv2.destroyAllWindows()
# End of Part 2

#Part 3
repeat = 3

tiledPic = np.tile(lgMapped, (repeat,1,1))
cv2.imshow('im',tiledPic)
cv2.waitKey(0)
cv2.destroyAllWindows()


xNew = np.tile(wx, (repeat,1))
yNew = cp.deepcopy(wy)
for i in range (1,repeat):
    yNew = np.concatenate((yNew, wy + 2*i*np.max(wy)))
z = xNew + 1j * yNew
# End of Part 3

# Part 4
alpha = np.arctan(np.log(r2/r1) / (2 * np.pi))
f = np.cos(alpha)
rz = z * f * np.exp(1j*alpha)
rz = np.exp(rz)
rotX = np.real(rz)
rotY = np.imag(rz)
rxMax = np.max(abs(rotX))
ryMax = np.max(abs(rotY))
xNew2 = (rotX / rxMax + 1) * len(img) / 2
yNew2 = (rotY / ryMax + 1) * len(img[0]) / 2
np.floor(xNew2,xNew2)
np.floor(yNew2,yNew2)
finalPic = np.zeros((len(img),len(img[0]),3),dtype='uint8')
xNew2 = xNew2.astype(int)
yNew2 = yNew2.astype(int)
np.clip(xNew2,0,len(img) - 1,xNew2)
np.clip(yNew2,0,len(img[0]) - 1,yNew2)
for p in range(len(xNew2)):
    for q in range(len(xNew2[0])):
        finalPic[yNew2[p][q]][xNew2[p][q]] = img[p%len(img)][q]

cv2.imshow('im',finalPic)
cv2.waitKey(0)
cv2.destroyAllWindows()
# End of Part 4
