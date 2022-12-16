import pickle
import pandas as pd 
import streamlit as st 

#read model 
model_kmeans = pickle.load(open('kmeans_model.pkl', 'rb')) 
pca = pickle.load(open('pca_model.pkl', 'rb'))

#judul web
st.title('Predict Cluster of Credit Card Customer')


st.markdown(
    "<p style='text-align: center;'>Made by <b><a href='https://www.linkedin.com/in/rosita-nurul-janatin-561145214/'>'Rosita Nurul Janatin</a></b> , <b><a href='https://www.linkedin.com/in/haikalefendi/'>'Haikal Efendi</a></b> & <b><a href='https://www.linkedin.com/in/ni-made-kirei-kharisma-handayani-90528b21a/'>Ni Made Kirei Kharisma Handayani</a></b></p>",
    unsafe_allow_html=True
)

st.image("https://www.offix.com/wp-content/uploads/2021/02/brick-and-mortar-bank-challenges.jpg")
#bagi kolom
col1, col2, col3 = st.columns(3)

with col1:
    BALANCE = st.number_input('Input nilai balance')
    BALANCE_FREQUENCY = st.number_input('Input nilai balance frequency')
    PURCHASES = st.number_input('Input nilai purchase')
    ONEOFF_PURCHASES = st.number_input('Input nilai one off purchase')
    INSTALLMENTS_PURCHASES = st.number_input('Input nilai installments purchases')
    CASH_ADVANCE = st.number_input('Input nilai cash advance')

with col2:
    PURCHASES_FREQUENCY = st.number_input('Input nilai purchases frequency')
    ONEOFF_PURCHASES_FREQUENCY = st.number_input('Input nilai one off purchases frequency')
    PURCHASES_INSTALLMENTS_FREQUENCY = st.number_input('Input nilai purchases installments frequency')
    CASH_ADVANCE_FREQUENCY = st.number_input('Input nilai cash advance frequency')
    CASH_ADVANCE_TRX = st.number_input('Input nilai cash advance trx')

with col3:
    PURCHASES_TRX = st.number_input('Input nilai purchases trx')
    CREDIT_LIMIT = st.number_input('Input nilai credit limit')
    PAYMENTS = st.number_input('Input nilai payments')
    MINIMUM_PAYMENTS = st.number_input('Input nilai minimum payments')
    PRC_FULL_PAYMENT = st.number_input('Input nilai prc full payments')
    TENURE = st.number_input('Input nilai tenure') 

feature = [[
            BALANCE,
            BALANCE_FREQUENCY,
            PURCHASES,
            ONEOFF_PURCHASES,
            INSTALLMENTS_PURCHASES,
            CASH_ADVANCE,
            PURCHASES_FREQUENCY,
            ONEOFF_PURCHASES_FREQUENCY,
            PURCHASES_INSTALLMENTS_FREQUENCY,
            CASH_ADVANCE_FREQUENCY,
            CASH_ADVANCE_TRX,
            PURCHASES_TRX,
            CREDIT_LIMIT,
            PAYMENTS,
            MINIMUM_PAYMENTS,
            PRC_FULL_PAYMENT,
            TENURE
            ]]

df = pd.DataFrame(feature, columns=['BALANCE',
                                    'BALANCE_FREQUENCY',
                                    'PURCHASES',
                                    'ONEOFF_PURCHASES',
                                    'INSTALLMENTS_PURCHASES',
                                    'CASH_ADVANCE',
                                    'PURCHASES_FREQUENCY',
                                    'ONEOFF_PURCHASES_FREQUENCY',
                                    'PURCHASES_INSTALLMENTS_FREQUENCY',
                                    'CASH_ADVANCE_FREQUENCY',
                                    'CASH_ADVANCE_TRX',
                                    'PURCHASES_TRX',
                                    'CREDIT_LIMIT',
                                    'PAYMENTS',
                                    'MINIMUM_PAYMENTS',
                                    'PRC_FULL_PAYMENT',
                                    'TENURE'])

#code untuk cluster
# num_clus = ''

df_pca = pca.transform(df)

#membuat tombol untuk cluster
if st.button('Cek Cluster'):
    cluster_result = str(model_kmeans.predict(df_pca)[0])
    cc_cluster = "Golongan customer : "+cluster_result
    st.success(cc_cluster)
   

   