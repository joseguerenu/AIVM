import streamlit as st 
import pandas as pd 
import matplotlib.pyplot as plt 
import geopy.distance
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
#from sklearn.metrics import r2_score

#leemos el csv y hacemos el tratamiento de nulos 
df = pd.read_csv('c:/Users/diazj/Documents/ProyectoML/opportunities_dataset_jf.csv') 
df.Estado.fillna('Nueva construcción', inplace = True)
df=df.fillna(0) 

#sacamos los elementos del formulario

st.title('AIVM | Artificial Intelligence Value Model') 
st.write('Rellena los campos a continuación:') 
latitud = st.number_input('Latitud',value=0.0000000,min_value=-180.0000000,max_value=180.0000000,step=1e-7,format="%.7f") 
longitud = st.number_input('Longitud',value=0.0000000,min_value=-180.0000000,max_value=180.0000000,step=1e-7,format="%.7f") 
superficie = st.number_input('Superficie m2',value=0.00,min_value=0.00,max_value=100000.00,step=1e-2,format="%.2f") 
estadoViviendaString = st.selectbox('Estado de la vivienda',['Segunda mano/buen estado','Segunda mano/para reformar','Vienda nueva construcción'])
tipoViviendaString = st.selectbox('Tipo de construcción',['Piso','Estudio','Atico','Dúplex','Chalet'])
ascensor = st.selectbox('Con/Sin ascensor', ['Con', 'Sin'])

#sacamos el tamaño del dataframe inicial 
st.write("Tamaño del dataset inicial:",len(df),"registros")
#borramos los registros del data set cuyo valor de TIPO VIVIENDA sea Torre o Finca
#puesto que para Torre sólo tenemos un registro y para Finca tenemos sólo 4 registros

df2=df.drop(df[df['Tipologia'] == 'Torre'].index)
df2=df2.drop(df2[df2['Tipologia'] == 'Finca'].index)

#ponemos como False el valor de has_elevator cuando no está informado
df2['Ascensor'] = df2['Ascensor'].replace({'False': 0, 'True': 1, 'Unknown': 0})
	
#borramos un outlier que dice que tiene una superficie de 16301 m2 y un precio de 1559400, 
#con lo cual el precio por m2 es de 96 euros
#que no es nada real y debe tratarse de un error
df2.drop(df2[df2['PrecioM2'] ==96].index, inplace=True)


#modificamos el dataframe para que el tipo de vivienda que es categórico
#pase a ser una variable numérica
df2['Estado'] = df2['Estado'].replace({"Segunda mano/para reformar":"Reformar", "Segunda mano/buen estado":"Bueno", "Nueva construcción":"Nueva"})
dummies = pd.get_dummies(df2['Estado'], drop_first = False)

# Añadimos las variables binarias al DataFrame
df2 = pd.concat([df2, dummies], axis = 1)

#str_A_reformar=0
#str_Buen_estado=0
#str_Nueva_construccion=0
#if estadoViviendaString=='Segunda mano/buen estado':
#	str_Buen_estado=1
#else:
#	if estadoViviendaString=='Segunda mano/para reformar':
#		str_A_reformar=1
#	else:
#		str_Nueva_construccion=1


#strPiso=0
#strEstudio=0
#strAtico=0
#strDuplex=0
#strChalet=0
#if tipoViviendaString=='Piso':
#	strPiso=1
#else:
#	if tipoViviendaString=='Estudio':
#		strEstudio=1
#	else:
#		if tipoViviendaString=='Ático':
#			strAtico=1
#		else:
#			if tipoViviendaString=='Dúplex':
#				strDuplex=1
#			else:
#				strChalet=1
					

strtieneascensor=0
if ascensor == 'Con':
	strtieneascensor=1

#funcion que calcula la distancia de cada uno de los anuncios al punto de coordenadas que hemos escogido en el formulario
def calculaDistancia(row):
    #R = 6378.0# km (radio de la Tierra ecuatorial)
    coords_1 = (row["Latitud"], row["Longitud"])
    coords_2 = (latitud, longitud)
    return geopy.distance.geodesic(coords_1, coords_2).km
df2["DistancePoints"] = df.apply(calculaDistancia, axis=1)

#el dataset para que el tipo de vivienda que es categórico
#pase a ser una variable numérica
df2['Tipologia'] = df2['Tipologia'].replace({"Ático":"Atico"})
dummies2 = pd.get_dummies(df2['Tipologia'], drop_first = False)
#no podemos poner el drop_first porque entonces elimina uno de los tipos del desplegable
# Añadimos las variables binarias al DataFrame
df2 = pd.concat([df2, dummies2], axis = 1)


#ahora filtramos en el dataset con los siguientes filtros
#Filtro 1: estado de la vivienda

if estadoViviendaString == 'Segunda mano/para reformar':
	df2=df2[df2.Reformar == 1]
else:
	if estadoViviendaString == 'Segunda mano/buen estado':
		df2=df2[df2.Bueno == 1]
	else:
		df2=df2[df2.Nueva == 1]


# Eliminamos la variable original Estado
df2 = df2.drop(columns=['Estado'])

#Filtro 2: tipo de la vivienda: que en la columna adecuada tenga informado un 1


if tipoViviendaString=='Piso':
	df2=df2[df2.Piso == 1]
else:
	if tipoViviendaString=='Estudio':
		df2=df2[df2.Estudio == 1]
	else:
		if tipoViviendaString=='Ático':
			df2=df2[df2.Atico == 1]
		else:
			if tipoViviendaString=='Dúplex':
				df2=df2[df2.Duplex == 1]
			else:
				if tipoViviendaString=='Chalet':
					df2=df2[df2.Chalet == 1]
					
# Eliminamos la variable original TIPO VIVIENDA
df2 = df2.drop(columns=['Tipologia'])

#Tercer filtro: sólo tenemos en cuenta los coincidentes en el valor de ascensor
df2=df2[df2.Ascensor == strtieneascensor]

st.write("Primera estimación: filtrando previamente para modelar sólo los datos del mismo estado de la vivienda el mismo tipo de construcción y si tiene o no ascensor")

st.write("Tamaño del dataset filtrado:",len(df2),"registros")
# Preparar los datos de entrenamiento 
X_train = df2[['Latitud', 'Longitud', 'MetrosConstruidos']] 
y_train = df2['Precio'] 
sizeDF=len(df2)

# Crear el modelo de regresión lineal 
modelo = LinearRegression() 
# Entrenar el modelo 
modelo.fit(X_train, y_train) 
# Predecir el precio utilizando los valores ingresados por el usuario 
X_pred = pd.DataFrame([[latitud, longitud, superficie]], columns=X_train.columns) 
precio_estimado = modelo.predict(X_pred)[0] 

st.write('El precio estimado haciendo uso de los',len(df2), 'registros del dataset de la vivienda es:', precio_estimado)
#preparamos el gráfico para pintarlo
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
x = np.array(df2['MetrosConstruidos'])
y = np.array(df2['Precio'])
m, b = np.polyfit(x, y , 1)
plt.plot(x, y, 'o')
#pintamos la línea de tendencia
plt.plot(x, m*x+b)
plt.xlabel("Superficie m2")
plt.ylabel("Precio €")
st.write(fig)

#sacamos las estadísticas de la regresión lineal
x = df2[['MetrosConstruidos']]
y = df2['Precio']
model = sm.OLS(y, x).fit()
st.write(model.summary())

#empezamos la segunda estimación
st.write("Segunda estimación: filtrando adicionalmente al filtro anterior las contrucciones que estén a una distancia inferior a 1KM")
df3=df2

#ahora ordenamos el df por la distancia
df3=df3.sort_values(by=['DistancePoints'], ascending=True, inplace=False)
#y ahora filtramos por los que tengan una distancia inferior a 1.0 km
df3=df3[df3.DistancePoints <= 1.0]

sizeDF3=len(df3)
st.write("Tamaño del dataset filtrado:",len(df3),"registros")
# Preparar los datos de entrenamiento 
X_train = df3[['Latitud', 'Longitud', 'MetrosConstruidos']] 
y_train = df3['Precio'] 
# Crear el modelo de regresión lineal 
modelo = LinearRegression() 
# Entrenar el modelo 
linear_regressor=modelo.fit(X_train, y_train) 
# Predecir el precio utilizando los valores ingresados por el usuario que no están ya filtrados en el Data Frame
X_pred = pd.DataFrame([[latitud, longitud, superficie]], columns=X_train.columns) 
precio_estimado = modelo.predict(X_pred)[0] 


st.write("El precio estimado haciendo uso de los",len(df3),"registros del dataset resultante del filtro es",precio_estimado," :sunglasses:")

#preparamos el gráfico para pintarlo
fig2 = plt.figure()
ax2 = fig2.add_subplot(1,1,1)

x = np.array(df3['MetrosConstruidos'])
y = np.array(df3['Precio'])
m, b = np.polyfit(x, y , 1)
plt.plot(x, y, 'o')
#pintamos la línea de tendencia
plt.plot(x, m*x+b)
plt.xlabel("Superficie m2")
plt.ylabel("Precio €")
st.write(fig2)

#sacamos las estadísticas de la regresión lineal
x = df3[['MetrosConstruidos']]
y = df3['Precio']
model = sm.OLS(y, x).fit()
st.write(model.summary())