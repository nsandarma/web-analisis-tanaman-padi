from . import app,render_template,request,df,pd,request,convert,redirect
from .routes import pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from .util import getScaler,getXandY
from .model import NeuralNetwork
from sklearn.preprocessing import MinMaxScaler
df = pd.read_csv('data_lengkap.csv')

class Neural_Network(object):
  def __init__(self):
    # inisisasi nilai input layer, bobot, dan output layer
    self.inputSize = 3
    self.outputSize = 1
    self.hiddenSize = 3
 
    #bobot
    self.W1 = np.random.randn(self.inputSize, self.hiddenSize)
    self.W2 = np.random.randn(self.hiddenSize, self.outputSize) 
 
  def forward(self, X):
    self.z = np.dot(X, self.W1) 
    self.z2 = self.sigmoid(self.z) 
    self.z3 = np.dot(self.z2, self.W2) 
    o = self.sigmoid(self.z3) 
    return o 
 
  def sigmoid(self, s):
    # fungsi aktivasi sigmoid
    return 1/(1+np.exp(-s))

  def sigmoidPrime(self, s):
    # fungsi derivatif sigmoid
    return s * (1 - s)

  def backward(self, X, y, o):
    self.o_error = y - o 
    self.o_delta = self.o_error*self.sigmoidPrime(o) 

    self.z2_error = self.o_delta.dot(self.W2.T) 
    self.z2_delta = self.z2_error*self.sigmoidPrime(self.z2) 
    
    # update nilai bobot
    self.W1 += X.T.dot(self.z2_delta)
    self.W2 += self.z2.T.dot(self.o_delta)

  def train (self, X, y):
    o = self.forward(X)
    self.backward(X, y, o)

@app.route('/backpropagation')
def backpropagation():
  return render_template('backprop.html',title='Backpropagation')

def hitung_prediksi(provinsi,X):
  prov = df[df['Provinsi'] == provinsi]
  X_test = X
  X_train = prov[['Luas Panen','Kelembapan','Suhu rata-rata']].values
  y_train = prov[['Produksi']].values
  X_scaler = X_train/np.amax(X_train, axis=0) 
  y_scaler = y_train/max(y_train)
  X_test_scaler = X_test/np.amax(X_test,axis=0)
  
  NN = Neural_Network()
  for i in range(1000):
    NN.train(X_scaler,y_scaler)
  hasil = NN.forward(X_test_scaler)
  
  return hasil*max(y_train)


@app.route('/prediksitahun',methods=['GET','POST'])
def prediksi_():
    if request.method == 'POST':
        provinsi = request.form['provinsi']
        X = np.random.rand(4,3)
        hasil = hitung_prediksi(provinsi,X)
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

    # X_test = np.array([[float(luas_panen),float(kelembapan),float(suhu_rata)]])
    # hasil = hitung_prediksi(provinsi,X_test)
    # hasil = f'{convert(hasil[0][0])} Kg / {convert(hasil[0][0] / 1000)} Ton'
    # return render_template('prediksi.html',prov=data_prov,data_mean=dmean,hasil=hasil)
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
  X_scaler = X.values/np.amax(X.values, axis=0) 
  y_scaler = y.values/max(y.values)

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
