import streamlit as st 
import pandas as pd 
import geopy.distance
from sklearn.linear_model import LinearRegression 
 
df = pd.read_csv('c:/Users/diazj/Documents/ProyectoML/20230502_20230505_scraping_idealista_postcode_FINAL_ii.csv') 
df.asset_condition.fillna('Nueva construcción', inplace = True)
df=df.fillna(0) 

#borramos los registros del data set cuyo valor de TIPO VIVIENDA sea Torre o Finca
#df2=df
df2=df.drop(df[df['TIPO VIVIENDA'] == 'Torre'].index)
df2=df2.drop(df2[df2['TIPO VIVIENDA'] == 'Finca'].index)

#ponemos como False el valor de has_elevator cuando no está informado
df2['has_elevator'] = df2['has_elevator'].replace({'False': 0, 'True': 1, 'No informado': 0})

var=pd.get_dummies(df2['asset_condition'],prefix='Asset status')

df2=pd.concat([df2,var], axis=1)

#preparamos el dataset para que el estado de la vivienda pueda ser un filtro
#asignando los valores del scale mapper

scale_mapper = {"Segunda mano/para reformar":2, "Segunda mano/buen estado":1, "Nueva construcción":0}
df2["Estadovivienda"] = df2["asset_condition"].replace(scale_mapper)
 
st.title('AIVM | Artificial Intelligence Value Model') 
st.write('Rellena los campos a continuación:') 
latitud = st.number_input('Latitud',value=0.0000000,min_value=-180.0000000,max_value=180.0000000,step=1e-7,format="%.7f") 
longitud = st.number_input('Longitud',value=0.0000000,min_value=-180.0000000,max_value=180.0000000,step=1e-7,format="%.7f") 
superficie = st.number_input('Superficie m2',value=0.00,min_value=0.00,max_value=100000.00,step=1e-2,format="%.2f") 
estadoViviendaString = st.selectbox('Estado de la vivienda',['Segunda mano/buen estado','Segunda mano/para reformar','Vienda nueva construcción'])
if estadoViviendaString=='Segunda mano/buen estado':
	estadoVivienda=1
else:
	if estadoViviendaString=='Segunda mano/para reformar':
		estadoVivienda=2
	else:
		estadoVivienda=0
#baños = st.number_input('Número de baños',value=1,step=0) 
tipoViviendaString = st.selectbox('Tipo de construcción',['Piso','Estudio','Ático','Dúplex','Chalet'])
strPiso=0
strEstudio=0
strAtico=0
strDuplex=0
strChalet=0
if tipoViviendaString=='Piso':
	strPiso=1
else:
	if tipoViviendaString=='Estudio':
		strEstudio=1
	else:
		if tipoViviendaString=='Ático':
			strAtico=1
		else:
			if tipoViviendaString=='Dúplex':
				strDuplex=1
			else:
				strChalet=1
					
ascensor = st.selectbox('Con/Sin ascensor', ['Con', 'Sin']) 
strhaselevator=False
if ascensor == 'Con':
	strhaselevator=True
	
def calculaDistancia(row):
    #R = 6378.0# km (radio de la Tierra ecuatorial)
    coords_1 = (row["lat"], row["lon"])
    coords_2 = (latitud, longitud)
    return geopy.distance.geodesic(coords_1, coords_2).km
df2["DistancePoints"] = df.apply(calculaDistancia, axis=1)

#el dataset para que el tipo de vivienda que es categórico
#pase a ser una variable numérica
dummies = pd.get_dummies(df2['TIPO VIVIENDA'], drop_first = False)
#no podemos poner el drop_first porque entonces elimina uno de los tipos del desplegable
# Añadimos las variables binarias al DataFrame
df2 = pd.concat([df2, dummies], axis = 1)
# Eliminamos la variable original TIPO VIVIENDA
df2 = df2.drop(columns=['TIPO VIVIENDA'])

#ahora filtramos en el dataset con los siguientes filtros
#Filtro 1: estado de la vivienda
df2=df2[df2.Estadovivienda == estadoVivienda]
#el valor de la distancia (que sea menos de 1 KM)
#Filtro 2: tipo de la vivienda: que en la columna adecuada tenga informado un 1

#df2=df2[df2.tipoViviendaString == 1]

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

#Tercer filtro: sólo tenemos en cuenta los coincidentes en el valor de has_elevator
df2=df2[df2.has_elevator == strhaselevator]


# Preparar los datos de entrenamiento 
X_train = df2[['lat', 'lon', 'built_space', 'Estadovivienda', 'has_elevator','Piso','Estudio','Ático','Dúplex','Chalet']] 
#X_train = df[['built_space']] 
y_train = df2['asset_price'] 
# Crear el modelo de regresión lineal 
modelo = LinearRegression() 
# Entrenar el modelo 
modelo.fit(X_train, y_train) 
# Predecir el precio utilizando los valores ingresados por el usuario 
X_pred = pd.DataFrame([[latitud, longitud, superficie, estadoVivienda, strhaselevator, strPiso,strEstudio,strAtico,strDuplex,strChalet]], columns=X_train.columns) 
#X_pred = pd.DataFrame([[ superficie]], columns=X_train.columns) 
precio_estimado = modelo.predict(X_pred)[0] 
sizeDF=len(df2)
st.write("Tamaño del dataset inicial:",len(df),"registros")
st.write("Primera estimación: filtrando previamente para modelar sólo los datos del mismo estado de la vivienda y mismo tipo de construcción")

st.write('El precio estimado haciendo uso de los',len(df2), 'registros del dataset de la vivienda es:', precio_estimado)

df3=df2

#ahora ordenamos el df por la distancia
df3=df3.sort_values(by=['DistancePoints'], ascending=True, inplace=False)
#y ahora filtramos por los que tengan una distancia inferior a 1.0 km
df3=df3[df3.DistancePoints <= 1.0]

sizeDF3=len(df3)

# Preparar los datos de entrenamiento 
X_train = df3[['lat', 'lon', 'built_space', 'Estadovivienda', 'has_elevator', 'Piso','Estudio','Ático','Dúplex','Chalet']] 

y_train = df3['asset_price'] 
# Crear el modelo de regresión lineal 
modelo = LinearRegression() 
# Entrenar el modelo 
modelo.fit(X_train, y_train) 
# Predecir el precio utilizando los valores ingresados por el usuario 
X_pred = pd.DataFrame([[latitud, longitud, superficie, estadoVivienda, strhaselevator, strPiso,strEstudio,strAtico,strDuplex,strChalet]], columns=X_train.columns)
#X_pred = pd.DataFrame([[40.4266011, -3.6324579, 72.0, 1, 1]], columns=X_train.columns) 
#X_pred = pd.DataFrame([[ superficie]], columns=X_train.columns) 
precio_estimado = modelo.predict(X_pred)[0] 

st.write("Segunda estimación: filtrando adicionalmente al filtro anterior las contrucciones que estén a una distancia inferior a 1KM")

st.write("El precio estimado haciendo uso de los",len(df3),"registros del dataset resultante del filtro es",precio_estimado)
