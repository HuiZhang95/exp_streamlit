# import the Streamlit library and the necessary data exploration and DataVizualization libraries.
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Create a dataframe called df to read the file train.csv.
df=pd.read_csv("train.csv")

# create 3 pages called "Exploration", "DataVizualization" and "Modelling" on Streamlit
st.title("Titanic : binary classification project")
st.sidebar.title("Table of contents")
pages=["Exploration", "DataVizualization", "Modelling"]
page=st.sidebar.radio("Go to", pages)


if page == pages[0]:
    st.write("### Presentation of data")

    # Display the first 10 lines of df on the web application Streamlit by using the method st.dataframe().
    st.dataframe(df.head(10))
    # Display informations about the dataframe on the Streamlit web application using the st.write() method in the same way as a print and the st.dataframe() method for a dataframe.
    st.write(df.shape)
    st.dataframe(df.describe())
    # Create a checkbox to choose whether to display the number of missing values or not, using the st.checkbox() method.
    if st.checkbox("Show NA"):
        st.dataframe(df.isna().sum())

if page == pages[1]:
    st.write("### DataVizualization")

    # Display in a plot the distribution of the target variable.
    # Note: To display a countplot on Streamlit, you must frame it with fig = plt.figure() and st.pyplot(fig).
    fig = plt.figure()
    sns.countplot(x = 'Survived', data = df)
    st.pyplot(fig)

    # Display plots to describe the Titanic passengers. Add titles to the plots.

    fig = plt.figure()
    sns.countplot(x = 'Sex', data = df)
    plt.title("Distribution of the passengers gender")
    st.pyplot(fig)
    fig = plt.figure()
    sns.countplot(x = 'Pclass', data = df)
    plt.title("Distribution of the passengers class")
    st.pyplot(fig)
    fig = sns.displot(x = 'Age', data = df)
    plt.title("Distribution of the passengers age")
    st.pyplot(fig)


    # analyse the impact of the different factors on the survival or not of passengers.
    # Display a countplot of the target variable according to the gender.
    # Display a plot of the target variable according to the classes.
    # Display a plot of the target variable according to the age.
    fig = plt.figure()
    sns.countplot(x = 'Survived', hue='Sex', data = df)
    st.pyplot(fig)
    fig = sns.catplot(x='Pclass', y='Survived', data=df, kind='point')
    st.pyplot(fig)
    fig = sns.lmplot(x='Age', y='Survived', hue="Pclass", data=df)
    st.pyplot(fig)

    # conclude the multivariate analysis by looking at the correlations between the variables.
    # Display the correlation matrix of the explanatory variables.

    fig, ax = plt.subplots()
    sns.heatmap(df.corr(), ax=ax)
    st.write(fig)


# Write "Modelling" at the top of the third page using the st.write() command in the Python script.
if page == pages[2] : 
    st.write("### Modelling")
    
    # remove the irrelevant variables (PassengerID, Name, Ticket, Cabin).
    df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

    # create a variable y containing the target variable. Create a dataframe X_cat containing the categorical explanatory variables and a dataframe X_num containing the numerical explanatory variables.
    y = df['Survived']
    X_cat = df[['Pclass', 'Sex',  'Embarked']]
    X_num = df[['Age', 'Fare', 'SibSp', 'Parch']]

    # replace the missing values for categorical variables by the mode and replace the missing values for numerical variables by the median.
    # encode the categorical variables.
    # concatenate the encoded explanatory variables without missing values to obtain a clean X dataframe.
    for col in X_cat.columns:
        X_cat[col] = X_cat[col].fillna(X_cat[col].mode()[0])
    for col in X_num.columns:
        X_num[col] = X_num[col].fillna(X_num[col].median())
    X_cat_scaled = pd.get_dummies(X_cat, columns=X_cat.columns)
    X = pd.concat([X_cat_scaled, X_num], axis = 1)


    # separate the data into a train set and a test set using the train_test_split function from the Scikit-Learn model_selection package.
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    # standardize the numerical values using the StandardScaler function from the Preprocessing package of Scikit-Learn.
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train[X_num.columns] = scaler.fit_transform(X_train[X_num.columns])
    X_test[X_num.columns] = scaler.transform(X_test[X_num.columns])

    # create a function called prediction which takes the name of a classifier as an argument and which returns the trained classifier.
    # Note: You can use LogisticRegression, SVC and RandomForestClassifier classifiers from the Scikit-Learn library for example.

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix
    def prediction(classifier):
        if classifier == 'Random Forest':
            clf = RandomForestClassifier()
        elif classifier == 'SVC':
            clf = SVC()
        elif classifier == 'Logistic Regression':
            clf = LogisticRegression()
        clf.fit(X_train, y_train)
        return clf
    
    def scores(clf, choice):
        if choice == 'Accuracy':
            return clf.score(X_test, y_test)
        elif choice == 'Confusion matrix':
            return confusion_matrix(y_test, clf.predict(X_test))


    # use the st.selectbox() method to choose between the RandomForest classifier, the SVM classifier and the LogisticRegression classifier. Then return to the Streamlit web application to view the select box.
    choice = ['Random Forest', 'SVC', 'Logistic Regression']
    option = st.selectbox('Choice of the model', choice)
    st.write('The chosen model is :', option)

    # The st.radio() method displays checkboxes to choose between many options. Copy the following code. Then save your finished Python script. Finally go back to the Streamlit web page and click on "Re-run" to get the new display. You can play with the selectbox and the checkboxes to see the classification results of the different models interactively.
    clf = prediction(option)
    display = st.radio('What do you want to show ?', ('Accuracy', 'Confusion matrix'))
    if display == 'Accuracy':
        st.write(scores(clf, display))
    elif display == 'Confusion matrix':
        st.dataframe(scores(clf, display))