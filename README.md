# Modelos-Inteligentes-de-Aprendizaje
Técnicas de reconocimiento de imagenes, a partir de Análisis de Discriminante Lineal (LDA) y Análisis de Discriminante Cuadrático (QDA).

<h2>Introducción</h2>
La presente investigación se desarrolla en dos partes:

PARTE UNO

Implementar los algoritmos de Discriminante Lineal y Discriminante Cuadrático. Para lo cual hay que:
1. Considerar el “saneamiento” de las matrices de covarianza utilizando el método de
Eigendecomposition.
2. Implementar la regularización de las matrices de covarianza para el caso cuadrático y lineal.


Reportar los hallazgos en el presente documento de resultados en PDF que detalle lo realizado.

PARTE DOS

Implementar un sistema de reconocimiento de imágenes. La tarea será reconocer entre rectángulos y óvalos. Los datos de entrenamiento y prueba son “círculos_cuadrados.zip”.<br>
1. Utilizar LDC.
2. Utilizar PCS como método de extracción de características.

Como nota, cabe recordar, que los cálculos y/o medidas que se hagan para training –i.e. matrices de
covarianza, medias, etc.- deberán ser usados para preprocesar los datos de test. No se deben
recalcular en los datos de test.

<h2>PARTE 1</h2>
Para el desarrollo de la presente investigación, en lo que corresponde a la primera parte, se utilizaron los datos de Wisconsin para la implementación de la Regresión Cuadrática y la Regresión Lineal.
<img width="621" height="180" alt="image" src="https://github.com/user-attachments/assets/e3f2ad96-8dbd-4af1-85b7-812d3be61da5" /><br>
Elección del porcentaje para training y test.<br>

Al ejecutar el programa, se pregunta al usuario qué porcentaje de training y test se desea, para las presentes pruebas se utilizó el porcentaje de 80% de training y 20% de test.
Mediante cálculos, se obtienen las vectores de clase 1 y 2 para los datos de training, así como sus índices:
<img width="391" height="119" alt="image" src="https://github.com/user-attachments/assets/9c3cb8e4-a08c-4496-b1a6-86bfd422ebfe" /><br>
Obtención de labels de clase 1 y 2, así como sus índices.

También se obtiene la matriz de datos de training y se separa en otras dos matrices, una para clase 1 y la otra para clase 2:<br>
<img width="398" height="109" alt="image" src="https://github.com/user-attachments/assets/7e7c9350-ebb5-42a6-8d3b-520799f5e9f8" /><br>
Obtención de matrices de datos de training.

Se realiza la clasificación a priori para c1 y c2:<br>
<img width="499" height="77" alt="image" src="https://github.com/user-attachments/assets/a18f1df2-740c-45a2-94e6-1a1362d89dc4" /><br>
Clasificación a Priori.

Se obtiene la matriz de covarianza del set de training, así como las matrices de covarianza de las sub matrices para clase 1 y clase 2 de training:<br>
<img width="432" height="130" alt="image" src="https://github.com/user-attachments/assets/a10e112c-ef3e-44f2-81f1-864dbf9e0c7a" /><br>
Obtención de las matrices de covarianza de training.

Mediante una función creada en matlab llamada “sanear()”, se sanean los valores negativos y se reconstruye la matriz.<br>
<img width="543" height="86" alt="image" src="https://github.com/user-attachments/assets/17b8331b-4aee-4bed-b52a-c1865a44f5ee" /><br>
Aplicación del “saneamiento” de las matrices de covarianza.

Se calculan también las etiquetas de test, así como el data set para test:<br>
<img width="410" height="81" alt="image" src="https://github.com/user-attachments/assets/40100002-ad8d-4c0d-b8df-9e39b7779f7b" /><br>
Calculo de etiquetas de test, así como del data set de test.

Se realiza el cálculo de QDC para la matriz de test, utilizando los valores de las matrices saneadas de covarianza de training de clase 1 y 2, así como los valores de las medias de c1 y c2 para training:<br>
<img width="1182" height="105" alt="image" src="https://github.com/user-attachments/assets/f2a0ad9b-0ba4-4171-86ed-0a8fe3b3b8d7" /><br>
Cálculo de QDC sin regularizar

Donde:<br>
* M_C1 y M_C2. Son las medias de las matrices de clase 1 y clase 2, respectivamente.<br>
* E_inv_C1 y E_inv_C2. Son las inversas de las matrices saneadas de clase 1 y 2.<br>
* E_C1 y E_C2. Son las matrices saneadas de clase 1 y 2.<br>
* class_priori_1 y class_priori_2, es la probabilidad de las clases a priori.<br>

Además, se calcula LDC:<br>
<img width="844" height="105" alt="image" src="https://github.com/user-attachments/assets/25d6639a-67a9-44e9-829e-10634cc93a1d"> <br>
Cálculo de LDC sin regularizar.

Donde:<br>
* M_C1 y M_C2. Son las medias de las matrices de clase 1 y clase 2, respectivamente.<br>
* inv(ssaneado). Es la inversa de la matriz completa saneada de training.<br>
* ssaneado. Es la matriz completa saneada de training.<br>
* class_priori_1 y class_priori_2, es la probabilidad de las clases a priori.<br>

En este punto, los valores de clase 1, se convertirán en clase 0; y los valores de clase 2, se convertirán en clase 1.
Con esto, se procede a calcular cuál clase es mayor, si QDC_c1 > QDC_c2, será clase 0, de lo contrario, será clase 1; lo mismo para LDC, se compara si LDC_c1 > LDC_c2, será clase 0, de lo contrario, será clase 1.<br>
<img width="541" height="301" alt="image" src="https://github.com/user-attachments/assets/56688bf2-95ca-4f99-842b-5201e9366704" /><br>
Estimación de la mayor clase.

Esto es, para poder comparar contra la clasificación de test y ver qué tan acertada es la tasa de reconocimiento tanto de QDC como de LDC.
Se calcula la tasa de reconocimiento de QDC y LDC sin regularizar. <br>
<img width="509" height="385" alt="image" src="https://github.com/user-attachments/assets/2f1c4a14-bb9f-4f4b-87ce-fdceb1d85721" /><br>
Obtención de la tasa de reconocimiento de QDC y LDC sin regularizar.

Ahora, se procede a Regularizar en QDC para clase 1 y clase 2, tomando como valor el rango de la variable “alpha”, que va desde 0 hasta 1, en intervalos de 0.1. <br>
<img width="1021" height="240" alt="image" src="https://github.com/user-attachments/assets/24ccd6e5-81d9-4bcf-b558-9fdb9ee9dd65" /><br>
Regularización de QDC.

Se clasifica QDC con los valores de test, pero en vez de 1 y 2, serán 0 y 1 respectivamente.<br>
<img width="393" height="224" alt="image" src="https://github.com/user-attachments/assets/100c9642-5aec-4659-9ca3-2607ee77d31d" /><br>
Clasificación de test para QDC.

Se calcula la tasa de reconocimiento de la matriz de QDC:<br>
<img width="533" height="176" alt="image" src="https://github.com/user-attachments/assets/a15ec34d-894e-4870-993e-f949fecb86e4" /><br>
Cálculo de la tasa de reconocimiento de QDC.

Por último, se grafica la tasa de reconocimiento de la matriz de QDC:<br>
<img width="408" height="121" alt="image" src="https://github.com/user-attachments/assets/dd58deb9-da93-4c1d-8d18-7725fcf220e8" /><br>
Se despliega la gráfica de la tasa de reconocimiento de QDC.

Se procede también a Regularizar en LDC para clase 1 y clase 2, tomando como valor el rango de la variable “gamma”, que va desde 0 hasta 1, en intervalos de 0.1. <br>
<img width="880" height="217" alt="image" src="https://github.com/user-attachments/assets/03802e3d-878e-4409-9d5e-be348289db80" /><br>
Regularización de LDC.

Se clasifica LDC con los valores de test, pero en vez de 1 y 2, serán 0 y 1 respectivamente.<br>
<img width="415" height="221" alt="image" src="https://github.com/user-attachments/assets/3673fab2-af33-4c8f-881a-183b3fdccb03" /><br>
Clasificación de test para LDC.

Se calcula la tasa de reconocimiento de la matriz de LDC:<br>
<img width="525" height="173" alt="image" src="https://github.com/user-attachments/assets/93b30de0-8fc4-49b4-b4f9-ff12cfb8bd66" /><br>
Cálculo de la tasa de reconocimiento de LDC.

Por último, se grafica la tasa de reconocimiento de la matriz de LDC:<br>
<img width="424" height="132" alt="image" src="https://github.com/user-attachments/assets/80c6a55b-6aa1-4bb7-9a39-a17df2dd93bb" /><br>
Se despliega la gráfica de la tasa de reconocimiento de LDC.

En este punto, también se despliega en pantalla la tasa de reconocimiento de QDC y de LDC sin regularizar.<br>
<img width="656" height="52" alt="image" src="https://github.com/user-attachments/assets/81e31a09-3bc5-4de1-a7bf-fcf731fb8b6a" /><br>
Tasa de reconocimiento de QDC y LDC sin regularizar.

<h2>PARTE 2</h2>
Para el desarrollo de esta actividad, se utilizan las imágenes de la carpeta de “circulos_cuadrados”, en donde las imágenes que tienen como prefijo “c”, son datos de training que para fines de la presente
prueba, equivalen a rectángulos; mientras que las imágenes con prefijo “o”, son datos de training que equivalen a óvalos; por otra parte, las imágenes con prefijo “t”, pertenecen a los datos de test, en ellas hay tanto rectángulos como óvalos.<br>
<img width="646" height="264" alt="image" src="https://github.com/user-attachments/assets/59c284a2-568c-4559-b1b7-de233b396ca9" /><br>
Carpeta de “círculos_cuadrados”

Se han separado de manera manual las imágenes correspondientes a test y training, y se han dispuesto en las siguientes carpetas: <br>
<img width="254" height="127" alt="image" src="https://github.com/user-attachments/assets/1d2671ab-4a3e-44e3-838c-494480a156f3" /><br>
Carpetas de test y training.

<img width="490" height="214" alt="image" src="https://github.com/user-attachments/assets/387a765c-ab35-411f-9586-09bf37bf93dd" /><br>
Imágenes dentro de la carpeta training.

<img width="468" height="125" alt="image" src="https://github.com/user-attachments/assets/5fca0822-9610-458e-a50b-0b7ac0fe48bb" /><br>
Imágenes dentro de la carpeta test.

En el código se procede a extraer las imágenes de la carpeta de training, y convertirlas a escala de grises, para poder utilizarlas mejor, después se convierten en tipo “double”. También se obtienen los
nombres de las imágenes dentro de la carpeta “training”, con el fin de poder transformar las que empiecen con “c” (rectángulos) en valores de 0, y las que empiecen con la letra “o” (óvalos) en 1, y almacenar en un vector de clases.<br>
<img width="643" height="304" alt="image" src="https://github.com/user-attachments/assets/64f9ac86-f973-4447-b87e-fbdbd3fdffdf" /><br>
Carga de imagen y conversión a escala de grises y “double” para Training.

<img width="376" height="143" alt="image" src="https://github.com/user-attachments/assets/ddaa8656-8974-4253-87e3-18ea62ce4b72" /><br>
Clasificación de los archivos de training, en un vector de clases de 0 y 1.

Se extraen ahora los datos de Test, y de igual forma, se convierten las imágenes en escala de grises, y en “double”, después se obtienen los nombres de las imágenes, los cuales servirán para mostrar en los resultados, de qué tipo de imagen se trata, de acuerdo a la predicción.<br>
<img width="695" height="239" alt="image" src="https://github.com/user-attachments/assets/e1061bc9-7781-47d3-9b27-5a5d1bfcb5b6" /><br>
Carga de imagen y conversión a escala de grises y “double” para Test.

Se aplica PCA a la matriz resultante de Training, para obtener los componentes principales.<br>
<img width="285" height="43" alt="image" src="https://github.com/user-attachments/assets/2d12c3f5-dc99-46d5-81ee-baa0c76578f9" /><br>
Obtención de PCA a partir de la Matriz resultante de Training.

También se multiplica el resultado de PCA por cada una de las matrices: de Training y de Test.<br>
<img width="339" height="72" alt="image" src="https://github.com/user-attachments/assets/41cb0153-027e-4124-867d-143b6ff7b310" /><br>
Producto de los Componentes Principales por la matriz resultante de Training y Test.

Se obtienen las etiquetas de clases y los índices para training, para clase 0 y 1.<br>
<img width="312" height="146" alt="image" src="https://github.com/user-attachments/assets/55fb94b0-e59c-43ac-a6d1-42992d053d30" /><br>
Clasificación de etiquetas e índices para training.

Después se calcula la probabilidad a priori de las clases 0 y 1, así como la obtención de los datos de training separados en clases, y la matriz de covarianza y su saneamiento.<br>
<img width="440" height="72" alt="image" src="https://github.com/user-attachments/assets/dda45d6a-2403-4474-860b-af70cc9d66af" /><br>
Clasificación a priori de training.

<img width="335" height="67" alt="image" src="https://github.com/user-attachments/assets/dd5163c6-b25b-4996-b4bd-61c0545bef80" /><br>
Separación de los datos de training en clases 0 y 1.

<img width="548" height="111" alt="image" src="https://github.com/user-attachments/assets/d858be50-b4a6-4af0-a039-f009108de600" /><br>
Obtención de la matriz de covarianza de todo el set de datos de training y su matriz saneada.

Se calcula el LDC con los datos de Test, las medias de training clasificadas, y las matrices saneadas. Después se clasifican los resultados de LDC, para ambos resultados de LDC, y se determina si “LDC_c1 > LDC_c2”, se clasifica como 0, de otra manera, se pone 1.<br>
<img width="811" height="230" alt="image" src="https://github.com/user-attachments/assets/9413aa6c-7079-4eb2-8a3a-0e546cce5245" /><br>
Cálculo de LDC por data set de la clase 0 y 1.

<img width="257" height="187" alt="image" src="https://github.com/user-attachments/assets/0d8f7e8f-1b65-4529-8f82-c77d561a3d7f" /><br>
Clasificación de los resultados de LDC.

En este punto se obtiene la tasa de reconocimiento, y se envía a pantalla el resultado de la predicción, arrojando mensajes si el archivo tal, de la carpeta de “test” corresponde a un rectángulo o a un óvalo.<br>
<img width="326" height="160" alt="image" src="https://github.com/user-attachments/assets/3fc5aa70-8dc7-426e-8319-6418fc95cc8e" /><br>
Obtención de la tasa de reconocimiento.

<img width="485" height="231" alt="image" src="https://github.com/user-attachments/assets/75f1ae19-fea8-4773-a9f9-91a8549034da" /><br>
Se envía a pantalla si la imagen se trata de un Rectángulo o de un Óvalo.

Por último, se grafican los resultados predichos contra los resultados reales.<br>
<img width="284" height="158" alt="image" src="https://github.com/user-attachments/assets/cf55f1a0-4620-4cd2-872f-b76cafabe673" /><br>
Valores predichos vs resultados reales a graficar.


<h2>Resultados</h2>
<b>Parte 1</b><br>
Con los resultados de la tasa de reconocimiento de QDC, se puede observar el porcentaje de reconocimiento respecto a cada uno de los valores que toma Alpha, siendo 0 con el que más se acerca al 100%.<br>
<img width="229" height="295" alt="image" src="https://github.com/user-attachments/assets/f5f2dfc5-214f-4a81-ae65-7dbb8632e145" /><br>
Tasa de reconocimiento de QDC

Con los resultados de la tasa de reconocimiento de LDC, se puede observar el porcentaje de reconocimiento respecto a cada uno de los valores que toma Gamma, siendo 1 con el que más se 15 acerca al 100%. Con esto se comprueba que el porcentaje de reconocimiento de Alpha cuando es 0, es igual al porcentaje de reconocimiento de Gamma cuando es 1. <br>
<img width="224" height="300" alt="image" src="https://github.com/user-attachments/assets/961161c7-7a41-4cc1-ab79-1e2259a0432c" /><br>
Tasa de reconocimiento de LDC

Los resultados de las tablas anteriores pueden ser expresados gráficamente de la siguiente manera:<br>
<img width="562" height="506" alt="image" src="https://github.com/user-attachments/assets/92641498-64ea-4a36-8bbb-dbc11ac20fda" /><br>
Tasa de Reconocimiento para QDC y LDC.

Mientras tanto, en consola se muestra el resultado del cálculo de las tasas de reconocimiento de QDC y LDC sin regularizar: <br>
<img width="527" height="40" alt="image" src="https://github.com/user-attachments/assets/58fa9691-18b0-45e7-86b6-bf05e9c2cf82" /><br>
Tasa de reconocimiento de QDC y LDC sin regularizar

<b>Parte 2</b><br>
Para los resultados de la parte 2, se obtuvo que al ejecutar LDC (sin regularizar), se obtuvo una tasa de reconocimiento del 100%, ya que todas las figuras de la carpeta de “test” fueron identificadas correctamente:<br>
<img width="318" height="107" alt="image" src="https://github.com/user-attachments/assets/29124993-60db-491d-ae57-189e6f0bfcf7" /><br>
Reconocimiento acertado de las imágenes.

Al graficar los valores predichos contra los valores reales, se observa que coinciden al 100%: <br>
<img width="562" height="506" alt="image" src="https://github.com/user-attachments/assets/83ca1953-0890-43a0-aa85-703735a64c21" /><br>
Gráfica del reconocimiento de las imágenes.

<h2>Conclusiones</h2>
<b>Parte 1</b><br>
* Se ha comprobado que los mayores índices de aproximación del valor predicho con respecto al valor conocido de test, se dan cuando, para QDC regularizado, alpha vale 0; y para cuando LDC
regularizado, gamma vale 1. De hecho, bajo estas condiciones, con alpha 0 y gamma 1, se obtiene el mismo porcentaje de desempeño.<br>
* Los valores de la matriz de LDC regularizada, cuando gamma vale 0, son valores NaN, por los que al clasificarse, como LDC_c1_end y LDC_c2_end tienen el mismo valor, son clasificados como 1, esto es debido a la condición especificada: LDC_c1_end(v, m)>LDC_c2_end(v, m); lo mismo sucede con los valores de QDC y LDC (regularizados y no regularizados) que ni uno ni otro es mayor, sino que ambos son iguales.<br>
* Se ha comprobado que, sin regularizar, la mayor tasa de reconocimiento se da con LDC, que coincide con el porcentaje que se da en alpha 0 y gamma 1, es decir, 97.3684 %.<br>

<b>Parte 2</b><br>
* Con PCA se han podido extraer los componentes principales de training, con los cuales se calculan las nuevas matrices para Training y Test. <br>
* Se comprueba que con LDC, se logra una excelente tasa de reconocimiento para este caso en concreto.












































