import pickle
import streamlit as st

model = pickle.load(open('prediksi-harga-mobil.sav', 'rb'))

st.title('prediksi-harga-mobil')
Engine_size = st.number_input('Input Engine_size')
Horsepower	 = st.number_input('Input Horsepower')
Wheelbase = st.number_input('Input Wheelbase ')
Width	= st.number_input('Width')
Length	= st.number_input('Length')
Fuel_capacity	= st.number_input('Fuel_capacity')
Fuel_efficiency	= st.number_input('Fuel_efficiency')

predict = ''

if st.button('predict'):
    predict = model.predict(
        [[Engine_size, Horsepower, Wheelbase, Width, Length, Fuel_capacity, Fuel_efficiency]]
    )
    st.write('Price : ', predict)
