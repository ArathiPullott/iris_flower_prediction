import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

st.write('''
# Simple iris flower prediction app
This will classify iris flowers
''')

st.sidebar.header('user input parameter')

def user_input_parameters():
    sepal_length=st.sidebar.slider('sepal length',4.3,7.9,5.4)
    sepal_width=st.sidebar.slider('sepal width',2.0,4.4,3.4)
    petalal_length = st.sidebar.slider('petal length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('petal width', 0.1, 2.5, 0.2)

    data={'sepal_length':sepal_length,
          'sepal_width':sepal_width,
          'petal_length':petalal_length,
          'petal_width':petal_width}
    features=pd.DataFrame(data,index=[0])
    return features
df=user_input_parameters()
st.subheader('user input features')
st.write(df)
iris=datasets.load_iris()
X=iris.data
Y=iris.target

clf=RandomForestClassifier()
clf.fit(X,Y)

##prediction
prediction=clf.predict(df)
prediction_proba=clf.predict_proba(df)

st.subheader('prediction')
st.write(iris.target_names[prediction])

st.subheader('prediction probability')
st.write(prediction_proba)