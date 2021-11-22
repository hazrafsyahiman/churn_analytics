import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("telecom_churn.csv")

st.header("Churn Analytics")
"""
This project will present the churn analytics which provide the insights of telecommunication industry churn data.
"""

st.sidebar.subheader('Dataset: ')
st.sidebar.write("<a href='https://www.kaggle.com/barun2104/telecom-churn'> Customer Churn </a>", unsafe_allow_html=True)

st.sidebar.header('Types of Insight')

option = st.sidebar.selectbox(
    'Select the insight type',
     ['Dataset','Heatmap','Decision Tree'])

if option=='Dataset':
    st.subheader('\n\nDataset')

    st.dataframe(df)

    st.subheader('\nColumn Unique Value')

    obj = ['Churn', 'AccountWeeks', 'ContractRenewal', 'DataPlan', 'DataUsage', 'CustServCalls', 'DayMins', 'DayCalls', 'MonthlyCharge', 
       'OverageFee', 'RoamMins']

    for column in obj:
      st.write(column, ':', df[column].unique(), '\n')

elif option=='Heatmap':
    st.subheader('\n\nCorrelation Heatmap')

    fig = plt.figure(figsize=(11,11))
    sns.heatmap(df.corr(), vmin=-1, vmax=1, annot=True, cmap="coolwarm")
    st.pyplot(fig)
    """
    Based on the heatmap, we can see that the top 3 features that affecting the customer to churn is customer service calls (**CustServCalls**), 
    average daytime minutes per month (**DayMins**) and recently renewed contract (**ContractRenewal**).
    """

elif option=='Decision Tree':
    st.subheader('\nDecision Tree')

    from sklearn import tree
    from sklearn.model_selection import train_test_split
    from sklearn import metrics
    import graphviz

    train_data = df.drop(['Churn'], axis = 1)
    train_label = df['Churn']

    def user_input_TestSize():
      return st.slider('Select the test dataset size', 0.1, 0.5, value = 0.2)

    def user_input_RandomState():
      return st.slider('Select the random state', 1, 100, value = 1)

    def user_input_MaxDepth():
      return st.slider('Select the max depth', 1, 100, value = 4)

    test_size = user_input_TestSize()
    test_size = user_input_RandomState()
    X_train, X_test, y_train, y_test = train_test_split(train_data, train_label, test_size = 0.2, random_state = 99, stratify = train_label)

    max_depth = user_input_MaxDepth()
    model = tree.DecisionTreeClassifier(max_depth = 4)
    model.fit(X_train, y_train);
    columns = list(train_data.columns)
    dot_data = tree.export_graphviz(model, out_file = None, feature_names = columns, class_names = ['No', 'Yes'], filled = True, rounded = True)
    y_pred = model.predict(X_test)

    st.write(f"Accuracy = {metrics.accuracy_score(y_test, y_pred):.5f}")
    st.write(f"Precision = {metrics.precision_score(y_test, y_pred):.5f}")
    st.write(f"Recall = {metrics.recall_score(y_test, y_pred):.5f}")
    st.write(f"F1 Score = {metrics.f1_score(y_test, y_pred):.5f}")