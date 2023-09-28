from . import app,render_template,request,df,pd,request,redirect,NeuralNetwork,getScaler
from .routes import pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from .util import getScaler,getXandY
from .model import NeuralNetwork
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
df = pd.read_csv('data-tanaman-padi-lengkap.csv')


@app.route('/backpropagation')
def backpropagation():
  return render_template('backprop.html',title='Backpropagation')

def hitung_prediksi_tahun(provinsi):

  prov = df[df['Provinsi'] == provinsi]
  X =  prov[['Luas Panen','Kelembapan','Suhu rata-rata',"Curah hujan"]].values
  y  =  prov[['Produksi']].values
  scalerX  = getScaler(X)
  scalerY = getScaler(y)
  X_scaler = scalerX.transform(X)
  y_scaler = scalerY.transform(y)
  X_train,X_test,y_train,y_test = train_test_split(X_scaler,y_scaler,random_state=42,test_size=0.12)
  
  NN = NeuralNetwork(input_size=4,hidden_size1=4,hidden_size2=4,output_size=1)
  NN.train(X_train,y_train,epochs=10000,learning_rate=0.1)
  hasil = NN.predict(X_test)
  
  return scalerY.inverse_transform(hasil)


@app.route('/prediksitahun',methods=['GET','POST'])
def prediksi_():
    if request.method == 'POST':
        
        provinsi = request.form['provinsi']
           
        hasil = hitung_prediksi_tahun(provinsi)
        X = ['2021','2022','2023','2024']
        hasil = [i[0] for i in hasil]
        hasil = pd.DataFrame({'Tahun':X,'Prediksi':hasil})

        fig = px.line(x = hasil['Tahun'], y =hasil['Prediksi'], template='plotly_dark', markers=True)
        fig.update_layout(title=f'Prediksi Produksi di Provinsi {provinsi}  Dari Tahun 2021-2024',
                   xaxis_title='Tahun',
                   yaxis_title='Jumlah Produksi')
        return fig.to_html()

    prov = df["Provinsi"].unique()
    return render_template('prediksi_.html',prov=prov,title='Prediksi 4 Tahun')

@app.route('/prediksi',methods=['GET','POST'])
def prediksi():
  data_prov = df.Provinsi.unique()
  if request.method == 'POST':
    provinsi = request.form['provinsi']
    return redirect(f'/prediksi/{provinsi}')
    
    
  return render_template('pred.html',prov=data_prov,title ="Prediksi")


@app.route('/prediksi/<provinsi>',methods=['GET',"POST"])
def prediksiProvinsi(provinsi):
  X,y =  getXandY(provinsi)
  scalerX = getScaler(X)
  scalerY = getScaler(y)
  X_scaler = scalerX.transform(X.values)
  y_scaler = scalerY.transform(y.values)
  rata_rata = X.mean().map("{:.2f}".format)
  NN = NeuralNetwork(input_size=4,hidden_size1=4,hidden_size2=4,output_size=1)
  NN.train(input_data=X_scaler,output_data=y_scaler,epochs=10000,learning_rate=0.1)


  if request.method == "POST":
    luas_panen = request.form['luas_panen']
    kelembapan = request.form['kelembapan']
    curah_hujan = request.form['curah_hujan']
    suhu_rata_rata = request.form['suhu_rata']
    test = pd.DataFrame({'Luas Panen':[luas_panen],'Curah hujan':[curah_hujan],'Kelembapan':[kelembapan],'Suhu rata-rata':[suhu_rata_rata]}).astype('int')
    XtestScaler = test.values/np.amax(X.values,axis=0)
    pred = NN.predict(XtestScaler)
    pred = pred*max(y.values)
    hasil  = str(round(pred[0][0],2))
    return render_template('prediksi.html',data_mean= rata_rata,hasil =hasil,provinsi=provinsi)
  return render_template('prediksi.html',data_mean=rata_rata,title='Prediksi',provinsi=provinsi)
