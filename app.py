from numpy.core.arrayprint import repr_format
import streamlit as st

import pickle
import numpy as np
model=pickle.load(open('Pickle_RL_CNC_lathe.pkl','rb'))


def predict_flank_wear_prediction(mp,doc,rpm):
    input=np.array([[mp,doc,rpm]]).astype(np.float64)
    pred=model.predict(input)
    return np.float(pred)

def main():
    
    html_temp = """
    <div style="background-color:#C0C817 ;padding:15px">
    <h2 style="font-family:Times New Roman;color:black;text-align:center;">CNC TOOL FLANK WEAR PREDICTION ML App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    #sep_len = st.text_input("Sepal Length","Type Here")
    mp=st.slider('Enter Machining Parameter_Feed rate (mm/min)',0.00,5.00)
    #sep_width = st.text_input("Sepal Width","Type Here")
    doc=st.slider('Enter Depth of cut (mm)',0.00,5.00)
    #petal_len = st.text_input("Petal Length","Type Here")
    rpm=st.slider('Enter Speed (rpm)',0.00,4000.00)
    

    
       

    if st.button("Predict"):
        output=predict_flank_wear_prediction(mp,doc,rpm)
        #st.success('The probability of this species is {}'.format(output))

        st.title('Approximate Value of Flank Wear will be {}'.format(output))
        

if __name__=='__main__':
    main()