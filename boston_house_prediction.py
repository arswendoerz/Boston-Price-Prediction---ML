import pickle
import numpy as np
import streamlit as st

input_data = (0.00632, 18.0,   2.31,   0.0,  0.538,  6.575,  65.2,  4.0900,  1.0,  296.0,    15.3 , 396.90 ,  4.98)
model = pickle.load(open('training_model.sav', 'rb'))

def predict_price(input_data):
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = model.predict(input_data_reshaped)
    return f"Prediksi harga rumah: {prediction[0]}"
    
def main():
    st.title("Prediksi Harga Rumah")

    crim = st.number_input('Masukkan nilai CRIM: ', value=None, step=None)
    zn = st.number_input('Masukkan nilai ZN: ', value=None, step=None)
    indus = st.number_input('Masukkan nilai INDUS: ', value=None, step=None)
    chas = st.number_input('Masukkan nilai CHAS: ', value=None, step=None)
    nox = st.number_input('Masukkan nilai NOX: ', value=None, step=None)
    rm = st.number_input('Masukkan nilai RM: ', value=None, step=None)
    age = st.number_input('Masukkan nilai AGE: ', value=None, step=None)
    dis = st.number_input('Masukkan nilai DIS: ', value=None, step=None)
    rad = st.number_input('Masukkan nilai RAD: ', value=None, step=None)
    tax = st.number_input('Masukkan nilai TAX: ', value=None, step=None)
    ptratio = st.number_input('Masukkan nilai PTRATIO: ', value=None, step=None)
    b = st.number_input('Masukkan nilai B: ', value=None, step=None)
    lstat = st.number_input('Masukkan nilai LSTAT: ', value=None, step=None)

    result = ''

    if st.button("Predict"):
        result = predict_price([crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, b, lstat])

    st.success(result)
main()

# cara run nya tulis ini di terminal tanpa tanda petik "streamlit run boston_house_prediction.py"