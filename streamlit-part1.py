import streamlit as st
import pandas as pd



from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
#import seaborn as sns
from sklearn import metrics
import category_encoders as ce
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB

header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()

st.markdown(
    """
    <style>
    .main {
        background-color: #F5F5F5;
    }
    """,
    unsafe_allow_html=True
)

@st.cache
def get_data(filename):
     income_data = pd.read_csv(filename)
     return income_data


with header:
    st.title('Welcome to my awesome Data Science project')
    st.text('In this project, I look into the transactions of NYC...')

with dataset:
    st.header('NYC taxi dataset')
    st.text('I found this dataset on blablabla.com..')

    income_data = get_data('D:\TM\Semester 1\AI\Lessons\Project\data\\adult_income_data.csv')
    #st.write(income_data.head())

    st.subheader('Education level distribution on the Income dataset')
    education_dist = pd.DataFrame(income_data['education'].value_counts())
    st.bar_chart(education_dist)

with features:
    st.header('This features I created')

    st.markdown('* **first feature:** I created this feature because of this...I calculated it using ...')
    st.markdown('* **second feature:** I created this feature because of this...I calculated it using ...')

with model_training:
    st.header('Time to train the model!')
    st.text('Here you get to choose the hyperparameters of the model and see how the performance change')

    sel_col, disp_col = st.columns(2)

    max_depth = sel_col.slider('What should be the max_depth of the model?', min_value=10, max_value=100, value=20, step=10)

    n_estimators = sel_col.selectbox('How many trees should there be?', options=[100,200,300,'No limit'], index = 0)

    sel_col.text('Here is a list of features in my data:')
    sel_col.write(income_data.columns)

    input_feature = sel_col._text_input('Which feature should be used as the input feature?','Sex')

    if n_estimators == 'Np limit':
        random_forest = RandomForestClassifier(max_depth= max_depth)
    else:
        random_forest = RandomForestClassifier(max_depth= max_depth, n_estimators=n_estimators)

    feature_names = list(income_data.columns.values)[:-1] 
    X = income_data.iloc[:, : -1]    #Assigning the feature 

    y = income_data[['Income']]  

    ce_ord = ce.OrdinalEncoder(cols = feature_names)
    X_TRANSFORM = ce_ord.fit_transform(X)
    
    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X_TRANSFORM, y, random_state=0, train_size = .70)

    random_forest.fit(X_train, y_train.values.flatten())
    prediction = random_forest.predict(X_test)
    #RF_Score = 
    disp_col.subheader('The accuracy of the model was calculated with the **score** metric:')
    disp_col.write(random_forest.score(X_test, y_test))

