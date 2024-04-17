import streamlit as st
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
 
# Download necessary NLTK resources
nltk.download('stopwords')
ps = PorterStemmer()
 
# Load the models
logistic_model = joblib.load('text_classifier.pkl')
svm_model = joblib.load('svm_classifier.pkl')
 
def clean_and_predict(text, model):
    # Text cleaning function
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
 
    # Predict using the selected model
    prediction = model.predict([review])
    return prediction
 
# User Interface 
st.title('Sentiment Analysis Tool')
model_choice = st.radio("Choose a model:", ('Logistic Regression', 'SVM'))
 
user_input = st.text_area("Enter text here:", value='', height=None, max_chars=None, key=None)
 
if st.button('Predict Sentiment'):
    # Choose model based on user input
    model_used = logistic_model if model_choice == 'Logistic Regression' else svm_model
    prediction = clean_and_predict(user_input, model_used)
    st.write('Predicted Sentiment:', prediction[0])
 
# show probabilities feature
if model_choice == 'Logistic Regression':
    if st.button('Show Prediction Probabilities'):
        probabilities = logistic_model.predict_proba([user_input])[0]
        st.write(f"Probability of Negative: {probabilities[0]*100:.2f}%")
        st.write(f"Probability of Positive: {probabilities[1]*100:.2f}%")

tab1, tab2 = st.tabs(["Dataset", "Functionality"])
tab1.write("The dataset comprises tweets that utilized the hashtag #TheSocialDilemma, reflecting a range of public opinions about the documentary. These tweets have been collected to provide insights into viewers' reactions, ranging from positive to negative sentiments. The dataset was specifically chosen to ensure a balanced perspective by including an equal number of positive and negative responses, which helps in training our models more effectively.")
tab2.write("""The tool processes user-inputted text (simulating tweet content) and predicts the sentiment as either "Positive" or "Negative." Users can choose between two machine learning models for the sentiment prediction:

**Logistic Regression:** Offers fast and interpretable results, ideal for linearly separable data.
**Support Vector Machine (SVM):** Utilizes a kernel trick to handle non-linear data, providing robust predictions even in complex scenarios.
""")
