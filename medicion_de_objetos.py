'''Calcula la distancia y el tamanio de un objeto
ocupando un laser apuntando al objeto y una cámara.

La relacion entre pixeles y medidas reales han sido calculadas para la camara empleada
'''

from skimage import io, color
import matplotlib.pyplot as plt
import numpy as np
from skimage import morphology
import math
from sklearn.cluster import KMeans

def lab_Kmeans(ima,nclases):#Aplica k means con el numero de clases indicado
    I=color.rgb2lab(ima)#    usando el modelo de color CIElab
    
    l=I[:,:,0]#Se aplica a cada capa
    a=I[:,:,1]
    b=I[:,:,2]
    
    L=l.reshape((-1,1))
    A=a.reshape((-1,1))
    B=b.reshape((-1,1))
    
    datos = np.concatenate((L,A,B),axis=1)
        
    clases=nclases
    salida=KMeans(n_clusters=clases).fit(datos)
        
    centros=salida.cluster_centers_
    etiquetas=salida.labels_
    
    for i in range(L.shape[0]):
        L[i]=centros[etiquetas[i]][0]
        A[i]=centros[etiquetas[i]][1]
        B[i]=centros[etiquetas[i]][2]
        
    L.shape=l.shape
    A.shape=a.shape
    B.shape=b.shape
    
    L=L[:,:,np.newaxis]
    A=A[:,:,np.newaxis]
    B=B[:,:,np.newaxis]
        
    k_lab =np.concatenate((L,A,B),axis=2)
    return k_lab


plt.close('all') 
imagen = io.imread('obj2_20cm.jpg')#imagen original

#CALCULO DE LA DISTANCIA DEL OBJETO APUNTADO CON EL LASER
lab = color.rgb2lab(imagen)#imagen en modelo de color cielab
#Se debe encontrar el laser, mas especificamente, su centro:
l = lab[:,:,0]#capa de luminosidad, debido al laser
r = imagen[:,:,0]#capa de tonos rojos, el laser es rojo

(f,c,capas) = imagen.shape
xc_img =int(f/2)#centro de la imagen
yc_img =int(c/2)

laser = np.zeros((f,c))
laser = np.where((l>80)&(r>245),1,0)#binarizacion de la luz del laser
#plt.figure('Laser'), plt.imshow(laser,cmap='gray')

pixeles_laserX= []
pixeles_laserY= []
for i in range(f):#Busca los pixeles blancos pertenecientes al laser
    for j in range(c):
        if laser[i,j]==1:
            pixeles_laserX.append(j)#guarda las posiciones
            pixeles_laserY.append(i)
xc_laser = int(sum(pixeles_laserX)/len(pixeles_laserX))#obtiene el promedio de los valores
yc_laser = int(sum(pixeles_laserY)/len(pixeles_laserY))

#Distancia euclidiana desde el centro de la imagen al punto del laser:
distancia = math.sqrt((xc_laser - xc_img)**2 + (yc_laser - yc_img)**2)

'''Ecuacion para calculo de profundidad, calculada a traves de una aproximacion polinomica
usando datos previos de la camara usada.(Pixel - cm)'''
profundidad = -3*10**(-5)*distancia**3 + 0.0272*distancia**2 - 8.2946*distancia + 870.9
print('profundidad',profundidad,'cm.')


#CALCULO DEL TAMANIO DEL OBJETO APUNTADO CON EL LASER
kmeans = lab_Kmeans(imagen, 2)#Separacion en dos clases: fondo y objeto
gris = color.rgb2gray(kmeans)
binarizada = np.where(gris>0.5,0,1)#binarizacion del objeto
# plt.figure('bina'), plt.imshow(binarizada,cmap='gray')

disco2 = morphology.disk(2)
sinRuido = morphology.erosion(binarizada,disco2)#Reduce ruido alrededor del objeto
disco1 = morphology.disk(1)
erosionB = morphology.erosion(sinRuido,disco1)
bordes = sinRuido - erosionB#resta de la erosion para obtener los bordes
plt.figure('bordes'), plt.imshow(bordes,cmap='gray')

#Se busca el primero y ultimo pixel perteneciente al objeto en ambos ejes X y Y
(px,py,px,py) = ([],[],[],[])
for i in range(f):
    for j in range(c):
        if bordes[i,j]==1:
            px.append(i)
            py.append(j)      

(minX, maxX) = (np.min(px),np.max(px))
(minY, maxY) = (np.min(py),np.max(py))

dx=maxX-minX
dy=maxY-minY

alto=(dx*profundidad)/520
ancho=(dy*profundidad)/520
print('alto: ',alto,'cm.')
print('ancho: ',ancho,'cm.')

