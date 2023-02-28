from __future__ import division, print_function

# Keras
from keras.models import load_model
from keras.applications.imagenet_utils import preprocess_input
from nilearn import datasets
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
plt.style.use('ggplot')

#nilearn - neuroimaging tailored library
from nilearn.input_data import NiftiMapsMasker
from nilearn import plotting

#sklearn - basic ML tools
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn import metrics

#keras - for NN models
from keras.models import Model, Sequential
from keras.layers import Input, Dense
from keras.layers import Dropout,GRU,LSTM
from keras import optimizers
#from keras.utils import plot_model
from keras import utils
from sklearn.metrics import roc_curve

# from keras.optimizers import SGD
from tensorflow.keras.optimizers import SGD
import math
from sklearn.metrics import mean_squared_error

#scipy- statistical analysis tools
from scipy.stats import ttest_1samp
from scipy import interp

from keras.applications.imagenet_utils import preprocess_input
from keras.models import load_model


from tensorflow.python import keras
import tensorflow as tf


# Flask 
from flask import Flask, request, render_template,redirect,url_for
from werkzeug.utils import secure_filename

import os
import re
import numpy as np
import cv2
import json
from zipfile import ZipFile
from flask_mysqldb import MySQL

# Definimos una instancia de Flask
app = Flask(__name__)

# conexion base de datos MYSQL
app.config['MYSQL_HOST']='localhost'
app.config['MYSQL_USER']='root'
app.config['MYSQL_PASSWORD']='modeldx123'
app.config['MYSQL_DB']='pacientestdah'
mysql=MySQL(app)


app.config['UPLOAD_FOLDER']="./uploads"


# Path del modelo preentrenado
path="C:/Users/Gianyosha/Desktop/API_DeepLearning-master/models/"

# Cargamos el modelo preentrenado


print('Modelo cargado exitosamente. Verificar http://127.0.0.1:5000/')
def preprocessing(func_file,confound_file,smith_atlas):  
  smith_atlas_rs_networks = smith_atlas.rsn70
  masker = NiftiMapsMasker(maps_img=smith_atlas_rs_networks,  # Smith stals
                         standardize=True, # centers and norms the time-series
                         memory='nilearn_cache', # cache
                         verbose=0) #do not print verbose
  
  all_subjects_data=[]
  time_series = masker.fit_transform(func_file, confounds=confound_file)    
  all_subjects_data.append(time_series)
  
  max_len_image=261
  #Reshaped
  all_subjects_data_reshaped=[]
  for subject_data in all_subjects_data:
    # Padding
    N= max_len_image-len(subject_data)
    padded_array=np.pad(subject_data, ((0, N), (0,0)), 
                        'constant', constant_values=(0))
    subject_data=padded_array
    subject_data=np.array(subject_data)
    subject_data.reshape(subject_data.shape[0],subject_data.shape[1],1)
    all_subjects_data_reshaped.append(subject_data)

  t_shape=np.array(all_subjects_data_reshaped).shape[1]
  RSN_shape=np.array(all_subjects_data_reshaped).shape[2]
  
  X_train=all_subjects_data_reshaped
  X_train = np.reshape(X_train, (len(X_train), t_shape, RSN_shape))

  X_train = X_train.astype('float32')

  return X_train

# Realizamos la predicción usando la imagen cargada y el modelo
def model_predict(file_path1,file_path2, path):
    smith_atlas = datasets.fetch_atlas_smith_2009()
    smith_atlas_rs_networks = smith_atlas.rsn70
    smith_atlas_rs_networks

    nii=file_path1
    cf=ffile_path2
    
    Xp=preprocessing(nii,cf,smith_atlas)
    model_load=keras.models.load_model(path+"model.h5")

    pred=model_load.predict(Xp)
    preds=[]
    for i in pred:
        for j in i:
            preds.append(j)
    
    return preds

@app.route('/', methods=['GET'])
def index():
    # Página principal
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        nombre = request.form['nombre']
        apellido = request.form['apellido']
        genero = request.form['genero']
        fecha = request.form['fecha']
        cedula = request.form['cedula']
        direccion = request.form['direccion']
        ciudad = request.form['ciudad']
        rh = request.form['rh']
        
        
        # imagen = request.form['file']

        # Obtiene el archivo del request
        f = request.files['file']
        nombreArchivo=secure_filename(f.filename)
        print(nombreArchivo)
        file_path1 = os.path.join(app.config['UPLOAD_FOLDER'],nombreArchivo)
        f.save(file_path1)

        f1 = request.files['file1']
        nombreArchivo1=secure_filename(f1.filename)
        print(nombreArchivo1)
        file_path2 = os.path.join(app.config['UPLOAD_FOLDER'],nombreArchivo1)
        f1.save(file_path2)
       

        preds = model_predict(file_path1,file_path2, path)
           # Enviamos el resultado de la predicción
        print('PREDICCIÓN',np.argmax(preds))
    
        preds = np.round(preds, decimals = 2)
        
        result = json.dumps(str(preds))    
        print(nombre)
        print(apellido)
        print(genero)
        print(fecha)
        print(cedula)
        print(direccion)
        print(ciudad)
        print(rh)
        print(f)
        print(preds) 
        diagnostico=" "
        cur = mysql.connection.cursor()
        cur.execute('INSERT INTO pacientes (nombre, apellido, genero, fecha,  cedula, direccion, ciudad, rh, imagen, predicion, diagnostico) VALUES (%s, %s, %s, %s, %s, %s, %s,%s,%s,%s,%s)',(nombre, apellido, genero, fecha, cedula, direccion, ciudad, rh, file_path1,result,diagnostico))
        mysql.connection.commit()   
        return result
    return redirect(url_for('validar'))
    
@app.route('/validar')
def validar():
    cur = mysql.connection.cursor()
    cur.execute('SELECT * FROM PACIENTES')
    data = cur.fetchall()


    return render_template("validar.html",pacientes=data)

@app.route('/diagnostico/<id>')
def diagnostico(id):
     cur = mysql.connection.cursor()
     cur.execute("SELECT * FROM PACIENTES WHERE id = %s",id)
     data = cur.fetchall()
     return render_template("update.html",paciente=data[0])

@app.route('/update/<id>',methods=['POST'])
def update_get(id):
     print(id)
     if request.method == 'POST':
        diag = request.form['diag']
        print(diag)
        cur = mysql.connection.cursor()
        cur.execute("""
        UPDATE pacientes  
        set diagnostico = %s
        WHERE id = %s
        """, (diag, id))
        mysql.connection.commit()   

     return redirect(url_for('validar'))

     
if __name__ == '__main__':
    app.run(debug=True, threaded=False)

