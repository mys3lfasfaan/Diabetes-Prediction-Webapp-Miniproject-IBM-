import pickle
import streamlit as st
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression

def main():
    path = 'finalized_model.sav'
    
    diabetes_model = pickle.load(open(path, 'rb'))
    
    # page title
    st.title('Diabetes Prediction using ML')
        
        
    # getting the input data from the user
    col1, col2, col3 = st.columns(3)
        
    with col1:
        Age = st.text_input('Age')
            
    with col2:
        Hypertension = st.text_input('Hypertension (0/1)')
        
    with col3:
        HeartDisease = st.text_input('Heart Disease (0/1)')
        
    with col1:
        bmi = st.text_input('BMI value')
        
    with col2:
        HbA1c = st.text_input('HbA1c Level')
        
    with col3:
        glucose = st.text_input('Glucose value')
        
        
    # code for Prediction
    diab_diagnosis = ''
        
    # creating a button for Prediction
    
    if st.button('Diabetes Test Result'):
        input_data = [int(Age),
                      int(Hypertension),
                      int(HeartDisease),
                      float(bmi),
                      float(HbA1c),
                      int(glucose)]
        data = np.asarray(input_data)
        data_reshaped = data.reshape(1,-1)
        diab_prediction = diabetes_model.predict(data_reshaped)
        diab_percentage = diabetes_model.predict_proba(data_reshaped)
        prob = np.max(diab_percentage, axis=1)
        max_prob = np.round(prob, 3)
    
        if (diab_prediction[0] == 1):
            diab_diagnosis = 'The person is diabetic. Estimated risk: {} %'.format(float(max_prob) *100)
            
        else:
            diab_diagnosis = 'The person is not diabetic '
        
    st.success(diab_diagnosis)

if __name__ == '__main__':
    main()
