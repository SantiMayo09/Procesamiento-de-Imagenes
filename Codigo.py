import cv2 as cv
import numpy as np
import random
import matplotlib.pyplot as plt

# Lectura de las imágenes
I1 = cv.imread('hombre1.jpg', 0)
I2 = cv.imread('hombre2.jpg', 0)
I3 = cv.imread('hombre3.jpg', 0)
I4 = cv.imread('hombre4.jpg', 0)
I5 = cv.imread('hombre5.jpg', 0)
I6 = cv.imread('hombre6.jpg', 0)
I7 = cv.imread('hombre7.jpg', 0)
I8 = cv.imread('hombre8.jpg', 0)
I9 = cv.imread('hombre9.jpg', 0)
I10 = cv.imread('hombre10.jpg', 0)
I11 = cv.imread('mujer1.jpg', 0)
I12 = cv.imread('mujer2.jpg', 0)
I13 = cv.imread('mujer3.jpg', 0)
I14 = cv.imread('mujer4.jpg', 0)
I15 = cv.imread('mujer5.jpg', 0)
I16 = cv.imread('mujer6.jpg', 0)
I17 = cv.imread('mujer7.jpg', 0)
I18 = cv.imread('mujer8.jpg', 0)
I19 = cv.imread('mujer9.jpg', 0)
I20 = cv.imread('mujer10.jpg', 0)

# Creación de la lista Poblacion
Poblacion = [I1, I2, I3, I4, I5, I6, I7, I8, I9, I10, I11, I12, I13, I14, I15, I16, I17, I18, I19, I20]

training= list(Poblacion)
training_hombres = training[:10]
training_mujeres = training[10:]
# Selección aleatoria de 6 índices sin reemplazo
indices_seleccionados = random.sample(range(len(Poblacion)), 6)

# Crear el vector prueba con las imágenes seleccionadas aleatoriamente
prueba = [Poblacion[i] for i in indices_seleccionados]

# Eliminar las imágenes seleccionadas de Poblacion
for imagen in prueba:
    for index, img in enumerate(training_hombres):
        if np.array_equal(imagen, img):
            del training_hombres[index]
            break  # Para evitar problemas eliminando elementos mientras se itera

for imagen in prueba:
    for index, img in enumerate(training_mujeres):
        if np.array_equal(imagen, img):
            del training_mujeres[index]
            break  # Para evitar problemas eliminando elementos mientras se itera

# Mostrar la población completa en la primera figura
plt.figure()
for idx, imagen in enumerate(Poblacion, 1):
    plt.subplot(5, 4, idx)
    plt.imshow(imagen, cmap='gray')
    plt.title(f'Población - Imagen {idx}')
    plt.axis('off')
plt.tight_layout()

# Mostrar las imágenes de prueba en la segunda figura
plt.figure()
for idx, imagen in enumerate(prueba, 1):
    plt.subplot(2, 3, idx)
    plt.imshow(imagen, cmap='gray')
    plt.title(f'Prueba - Imagen {idx}')
    plt.axis('off')
plt.tight_layout()


# Mostrar las imágenes de entrenamiento hombres en la tercera figura
plt.figure()
for idx, imagen in enumerate(training_hombres, 1):
    plt.subplot(5, 2, idx)
    plt.imshow(imagen, cmap='gray')
    plt.title(f'Training  hombres- Imagen {idx}')
    plt.axis('off')
plt.tight_layout()

# Mostrar las imágenes de entrenamiento mujeres en la tercera figura
plt.figure()
for idx, imagen in enumerate(training_mujeres, 1):
    plt.subplot(5, 2, idx)
    plt.imshow(imagen, cmap='gray')
    plt.title(f'Training Mujeres - Imagen {idx}')
    plt.axis('off')
plt.tight_layout()
plt.show()


areas_h= []
per_h=[]
comp_h=[]
red_h=[]
kernel = np.ones((5, 5), np.uint8)
elo_h=[]
for imagen in training_hombres:
    _, binarizada = cv.threshold(imagen, 215, 256, cv.THRESH_BINARY)
    invertida = cv.bitwise_not(binarizada)  # Invierte la imagen binarizada
    dila = cv.dilate(invertida, kernel)
    area = np.sum(dila == 255)
    areas_h.append(area)
    erosion = cv.erode(dila, kernel)
    frontera = dila - erosion
    peri= np.sum(frontera == 255)
    per_h.append(peri)   
    compacidad= (peri**2)/area
    comp_h.append(compacidad)
    redondez= (4*np.pi)*(area/peri**2)
    red_h.append(redondez)
    
    contornos, _ = cv.findContours(frontera, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    elipse = cv.fitEllipse(contornos[0])
    (x, y), (eje_mayor, eje_menor), angulo = elipse
    M = max(eje_mayor, eje_menor)
    N = min(eje_mayor, eje_menor)
    Elongacion= area/(M*N)
    elo_h.append(Elongacion)
    print(area)
    print(peri)
    print(compacidad)
    print(redondez)
    print(Elongacion)


areas_m= []
per_m=[]
comp_m=[]
red_m=[]
elo_m=[]
for imagen in training_mujeres:
    _, binarizadam = cv.threshold(imagen, 215, 256, cv.THRESH_BINARY)
    invertidam = cv.bitwise_not(binarizadam)  # Invierte la imagen binarizada
    dilam = cv.dilate(invertidam, kernel)
    aream = np.sum(dilam == 255)
    areas_m.append(aream)
    erosionm = cv.erode(dilam, kernel)
    fronteram = dilam - erosionm
    perim= np.sum(fronteram == 255)
    per_m.append(perim)   
    compacidadm= (perim**2)/aream
    comp_m.append(compacidadm)
    redondezm= (4*np.pi)*(aream/perim**2)
    red_m.append(redondezm)
    
    contornosm, _ = cv.findContours(fronteram, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    elipsem = cv.fitEllipse(contornosm[0])
    (x, y), (eje_mayorm, eje_menorm), angulom = elipsem
    Mm= max(eje_mayorm, eje_menorm)
    Nm = min(eje_mayorm, eje_menorm)
    Elongacionm= aream/(Mm*Nm)
    elo_m.append(Elongacionm)
    print(aream)
    print(perim)
    print(compacidadm)
    print(redondezm)
    print(Elongacionm)


promedio_areas_m = np.mean(areas_m)
promedio_per_m = np.mean(per_m)
promedio_comp_m = np.mean(comp_m)
promedio_red_m = np.mean(red_m)
promedio_elo_m = np.mean(elo_m)

promedio_areas_h = np.mean(areas_h)
promedio_per_h = np.mean(per_h)
promedio_comp_h = np.mean(comp_h)
promedio_red_h = np.mean(red_h)
promedio_elo_h = np.mean(elo_h)

hombre=[]
mujer=[]
for imagen in prueba:
    _, binarizadap = cv.threshold(imagen, 215, 256, cv.THRESH_BINARY)
    invertidap = cv.bitwise_not(binarizadap)  # Invierte la imagen binarizada
    dilap = cv.dilate(invertidap, kernel)
    areap = np.sum(dilap == 255)
    erosionp = cv.erode(dilap, kernel)
    fronterap = dilap - erosionp
    perip= np.sum(fronterap == 255)
    compacidadp= (perip**2)/areap
    redondezp= (4*np.pi)*(areap/perip**2)
    contornosp, _ = cv.findContours(fronterap, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    elipsep = cv.fitEllipse(contornosp[0])
    (x, y), (eje_mayorp, eje_menorp), angulom = elipsem
    Mp= max(eje_mayorp, eje_menorp)
    Np = min(eje_mayorp, eje_menorp)
    Elongacionp= areap/(Mp*Np)
    print(aream)
    print(perim)
    print(compacidadm)
    print(redondezm)
    print(Elongacionm)
    es_mujer=np.sqrt((promedio_areas_h-areap)**2+(promedio_per_h-perip)**2+(promedio_elo_h-Elongacionp)**2+(promedio_comp_h-compacidadp)**2+(promedio_red_h-redondezp)**2)
    es_hombre=np.sqrt((promedio_areas_m-areap)**2+(promedio_per_m-perip)**2+(promedio_elo_m-Elongacionp)**2+(promedio_comp_m-compacidadp)**2+(promedio_red_m-redondezp)**2)
    print("distancia hombre")
    print(es_hombre)
    print("distancia mujer")
    print(es_mujer)
    if es_hombre<es_mujer:
         hombre.append(imagen)         
    else:
         mujer.append(imagen)


plt.figure()
for idx, imagen in enumerate(hombre, 1):
    plt.subplot(3, 2, idx)
    plt.imshow(imagen, cmap='gray')
    plt.title(f'Hombre - Imagen {idx}')
    plt.axis('off')
plt.tight_layout()

plt.figure()
for idx, imagen in enumerate(mujer, 1):
    plt.subplot(3, 2, idx)
    plt.imshow(imagen, cmap='gray')
    plt.title(f'Mujer - Imagen {idx}')
    plt.axis('off')
plt.tight_layout()

plt.show()
