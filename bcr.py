from copyreg import pickle
import streamlit as st
import numpy as np
from numpy import array
from numpy import argmax
from numpy import genfromtxt
import pandas as pd
import shap
from sklearn import svm
from sklearn.svm import SVC
import matplotlib.pyplot as plt

st.set_page_config(page_title="Probability prediction of breast cancer recurrence risk", layout="centered")
plt.style.use('default')

df=pd.read_csv('modified_train_data_min.csv',encoding='utf8')
trainy=df.Recurrence_after_2_years
trainx=df.drop('Recurrence_after_2_years',axis=1)
svm=SVC(kernel='rbf',
        C=1,
        gamma=0.05,
        decision_function_shape='ovo',
        verbose=3,
        probability = True,
        random_state=1)
svm.fit(trainx, trainy)