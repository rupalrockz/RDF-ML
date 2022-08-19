import streamlit as st
import pandas as pd
import numpy as np
import rdflib
import time


def main():

    st.title('RDF-ML')
    st.title('A web based SPARQL Processing Tool for RDF Data & Applying Machine Learning')
    uploaded_file = st.file_uploader(label='Upload your RDF file', type='nt')

    q = st.text_area(label='Enter the SPARQL Query', height=250)

    btn = st.button('Execute')

    if uploaded_file is not None and q != '':

        g = rdflib.Graph()
        g.parse(uploaded_file, format="nt")

        len(g)

        ls = []
        cn = column_names(q)
        start=time.perf_counter()

        for row in g.query(q):
            temp = []
            temp = [float(row[i]) for i in range(len(cn))]
            ls.append(temp)

        end=time.perf_counter()
        df = pd.DataFrame(ls)

        df.columns = cn

    if btn:

        st.dataframe(df)
        st.write('Time Taken by the SPARQL Query ',(end-start))
       
        st.download_button(label="Download result to CSV",data=df.to_csv(),file_name='large_df.csv',mime='text/csv')

        #--------------------------------------------------------
        x=df.iloc[:,0:8]
        y=df.iloc[:,8]
        m=len(y)
        from sklearn.model_selection import train_test_split
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)

        from sklearn.neural_network import MLPClassifier

        #-----------------------------------

        from sklearn.linear_model import LogisticRegression
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.svm import SVC
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.naive_bayes import GaussianNB
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score,confusion_matrix
        from sklearn.metrics import roc_auc_score
        from sklearn.metrics import roc_curve
        from sklearn.metrics import f1_score
        from sklearn.metrics import classification_report
        features=['Glucose','DiabetesPedigreeFunction','Insulin','Age','BloodPressure']
        x=df[features]   
        y=df['Outcome']
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)

        #option = st.selectbox('Select Any ML Model?',('Random Forest', 'Logistic Regression','Decision Tree'))
        tab1, tab2, tab3,tab4 = st.tabs([ 'Logistic Regression','Random Forest','Decision Tree','Multi-Layer-Perceptron'])

        #st.write('You selected:', option)
        with tab1:
            st.subheader('Logistic Regression')
            lm=LogisticRegression()
            lm.fit(x_train,y_train)
            yhat_lm=lm.predict(x_test)
            lm_score=f1_score(y_test,yhat_lm)
            lm_accuracy=accuracy_score(y_test,yhat_lm)
            st.write('Logistic Regression Score',lm_score)
            st.write('Logistic Regression Accuracy',lm_accuracy)
            st.write(classification_report(y_test,yhat_lm))
            
        with tab2:
            st.subheader('Random Forest')
            random=RandomForestClassifier()
            random.fit(x_train,y_train)
            yhat_random=random.predict(x_test)
            random_score=f1_score(y_test,yhat_random)
            random_accuracy=accuracy_score(y_test,yhat_random)
            st.write('Random Forest Score',random_score)
            st.write('Random Forest Accuracy',random_accuracy)
            st.write(classification_report(y_test,yhat_random))
            

        with tab3:
            st.subheader('Decision Tree')
            tree=DecisionTreeClassifier()
            clf=tree.fit(x_train,y_train)
            yhat_tree=tree.predict(x_test)
            tree_score=f1_score(y_test,yhat_tree)
            tree_accuracy=accuracy_score(y_test,yhat_tree)
            st.write('Decision Tree Score',tree_score)
            st.write('Decision Tree Accuracy',tree_accuracy)
            st.write(classification_report(y_test,yhat_tree))

        with tab4:
            st.subheader('MLP with hiddenlayer=10 & max iterations = 1500')
            mlp=MLPClassifier(hidden_layer_sizes=(10),max_iter=1500)
            mlp.fit(x_train,y_train)

            y_pred=mlp.predict(x_test)

            from sklearn import metrics
            cnf_matrix=metrics.confusion_matrix(y_test,y_pred)
            st.write(cnf_matrix)
            m_accuracy=[]
            m_precision=[]
            m_recall=[]
            m_r2score=[]

            st.write("Accuracy :",metrics.accuracy_score(y_test,y_pred))
            st.write("Precision :",metrics.precision_score(y_test,y_pred))
            st.write("Recall :",metrics.recall_score(y_test,y_pred))



def column_names(s):

    l = []
    lines = s.splitlines()
    for line in lines:

        if line.find('SELECT') != -1:
            l = line.split('?')
            l.pop(0)
            break

    cn = [i.strip() for i in l]  # cn - column names
    return cn


if __name__ == "__main__":

    main()





