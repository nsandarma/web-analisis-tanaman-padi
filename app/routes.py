from app import app,redirect,render_template,send_from_directory,send_file,request
import pandas as pd
import numpy as np
from babel.numbers import format_decimal


df = pd.read_csv('data-tanaman-padi-lengkap.csv')
df_de = df.copy()

def convert(number):
    return format_decimal(number)

@app.route('/')
def home():
    title = 'Dashboard'
    return render_template('index.html',len_data=len(df),title=title)


@app.route('/table')
def table():
    return render_template('tables.html',data=df,total=len(df),title='Dataset')
    # return render_template_string(df.to_html())
@app.route('/analisis_data_provinsi/<provinsi>')
def provinsi(provinsi):
    if provinsi == 'sumut':
        data = df[df['Provinsi'] == 'Sumatera Utara']
        prov = "Sumatera Utara"
        pages = ['pages/curah_hujan_1.html','pages/kelembapan_1.html','pages/suhu_rata-rata.html','pages/luas_panen.html']
    else:
        data = df[df['Provinsi'] == 'Kepulauan Riau']
        prov = "Kepulauan Riau"
        pages =  ['curah_hujan_2.html','kelembapan_2.html','suhu_rata-rata_2.html','luas_panen_2.html'  ]
        pages =  ['grafik/curah_hujan_terendah.html','grafik/kelembapan_terendah.html','grafik/suhu_terendah.html','grafik/luas_panen_terendah.html']
    info = {'mean_luas_panen':int(data['Luas Panen'].mean()),'max_luas_panen':int(data['Luas Panen'].max()),'min_luas_panen':int(data['Luas Panen'].min()),'mean_curah_hujan':int(data['Curah hujan'].mean()),'maks_curah_hujan':int(data['Curah hujan'].max()),'min_curah_hujan':int(data['Curah hujan'].min()),'mean_kelembapan':int(data['Kelembapan'].mean()),'min_kelembapan':int(data['Kelembapan'].min()),'maks_kelembapan':int(data['Kelembapan'].max()),'mean_suhu':int(data['Suhu rata-rata'].mean()),'maks_suhu':int(data['Suhu rata-rata'].max()),'min_suhu':int(data['Suhu rata-rata'].min())}
    return render_template('provinsi_1.html',data=data,info=info,pages=pages,title=f'Analisis Data {provinsi}',prov=prov)
    
    
@app.route('/report',methods=['GET'])
def report():
    return redirect('/prediksi')
    

@app.route('/table/<provinsi>')
def table_provinsi(provinsi):
    pr = str(provinsi).title()
    
    if provinsi:
        if provinsi == 'total_produksi':
            provinsi = {}
            

            for p in df_de.Provinsi.unique():
                produksi = df_de[df_de['Provinsi'] == p]['Produksi']
                rata_rata = np.mean(produksi)
                total_produksi = sum(produksi)

                provinsi[p] = [rata_rata,total_produksi] 

            
            return render_template('tables_total_produksi.html',data=provinsi,title="Total Produksi") 
        else:
            data = df[df['Provinsi'] == pr]
            return render_template('tables.html',data=data,total=len(data),title=provinsi)
    else:
        "Provinsi Tidak ditemukan"

    
