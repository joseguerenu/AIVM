# APP DESCRIPCIÓN
El objetivo de la APP es realizar un estimador de precio de una vivienda por medio de un tratamiento de datos matemáticos y estadísticos a un data set 
de anuncios de ventas de construcciones de las ciudades Madrid y Barcelona obtenidos en internet con datos entrenados con un modelo de regresión lineal. 
La APP toma como datos de input 6
variables que son las siguientes:\
-Coordenadas geográficas Latitud\
-Coordenadas geográficas Longitud\
-Superficie en metros cuadrados\
-Estado de la vivienda\
-Tipo de construcción\
-Con o sin ascensor\
La aplicación hace dos estimaciones entrenando dos data set distintos que son los del resultado de filtrar sólo los anuncios que cumplen las siguientes
condiciones\
-Primera estimación. El data set se filtra por los valores elegidos por el usuario en los tres últimos desplegables (estado de la vivienda, tipo de 
construcción y con o sin ascensor)\
-Segunda estimación. El data set se filtra por los mismos criterios de la estimación anterior y adicionalmente se añade el filtro de que sólo se tienen 
en cuenta en el entrenamiento del modelo y la estimación aquellas viviendas que estén geográficamente situadas a menos de 1 km de las coordenadas geográficas
introducidas en el formulario en los dos primeros campos


![APP_AIVM](https://github.com/joseguerenu/AIVM/assets/136623944/6efc8f91-d9fe-4bad-a610-0b9a61cad779)
