# Importing Required Library
import streamlit as st # type: ignore
import pandas as pd # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.svm import SVC # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore
from sklearn.pipeline import make_pipeline # type: ignore
from sklearn import metrics # type: ignore
import nltk # type: ignore
import re
from nltk.corpus import stopwords # type: ignore
from nltk.tokenize import word_tokenize # type: ignore
from nltk.stem import WordNetLemmatizer # type: ignore
from imblearn.over_sampling import SMOTE # type: ignore
from sklearn.metrics import classification_report, accuracy_score # type: ignore
from sklearn.metrics import ConfusionMatrixDisplay # type: ignore
from sklearn.preprocessing import LabelEncoder # type: ignore
from sklearn.metrics import confusion_matrix # type: ignore
from langchain_google_genai import ChatGoogleGenerativeAI # type: ignore
from langchain_core.prompts import ChatPromptTemplate # type: ignore

# Downloading for Stopwords Process
nltk.download('punkt') # type: ignore
nltk.download('wordnet') # type: ignore
nltk.download('stopwords') # type: ignore
nltk.download('punkt_tab') # type: ignore

# API Google Generative AI Configuration
API_KEY = "AIzaSyDh7sAqxYIUzWajFP2jC5i95uFq_CgE-_4"  # Ganti dengan API key Anda

def chat(contexts, history, question):
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.7,
        api_key=API_KEY
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a sentiment analysis expert assistant. Your role is to explain and provide insights about sentiment classes and topics discussed in each sentiment class.",
            ),
            (
                "human",
                "Here is the sentiment data: {contexts}\n"
                "Use this chat history to generate relevant answers from recent conversations: {history}\n"
                "The user question is specifically related to sentiment analysis and topic extraction: {question}"
            ),
        ]
    )

    chain = prompt | llm
    completion = chain.invoke(
        {
            "contexts": contexts,
            "history": history,
            "question": question,
        }
    )

    answer = completion.content
    input_tokens = completion.usage_metadata['input_tokens']
    completion_tokens = completion.usage_metadata['output_tokens']

    result = {}
    result["answer"] = answer
    result["input_tokens"] = input_tokens
    result["completion_tokens"] = completion_tokens
    return result

# Streamlit App Title
st.title("Social Media Post Sentiment Analysis Classification and Topic Extraction")

# File Upload for Social Media Posts
uploaded_file = st.file_uploader("Upload your CSV file with social media posts", type=["csv"])

if uploaded_file:
    # Load data
    df = pd.read_csv(uploaded_file)

    # Data cleaning
    st.write("### Data Preprocessing")
    df = df[["text", "sentiment"]]

    # Handle missing values
    df.dropna(subset=['text'], inplace=True)
    # Clean text data
    df['text'] = df['text'].str.replace(r'[^a-zA-Z\s]', '', regex=True).str.lower()
    # Remove duplicates
    df.drop_duplicates(inplace=True)

    # Lemmatizer and Stopwords Process
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    def preprocess_text(text):
        tokens = word_tokenize(text.lower())  # Tokenisasi dan konversi ke huruf kecil
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]  # Lemmatization dan hapus stopwords
        return ' '.join(tokens)

    df['text'] = df['text'].apply(preprocess_text)

    # Missing Value
    df['text'].replace("", np.nan, inplace=True)
    df['text'] = df['text'].fillna(df['text'].mode()[0])

    # Display raw data
    st.write("#### Raw Dataset")
    st.dataframe(df.head())

    # Sentiment Analysis
    st.write("### Sentiment Analysis")
    if 'text' in df.columns:
        # change the 'sentiment' column to numeric
        le = LabelEncoder()
        df['sentiment'] = le.fit_transform(df['sentiment']) # merubah tipe sentiment menjadi numerik --> 0 : negative, 1 : neutral, 2 : positive

        # Vectorization Process
        vectorizer = TfidfVectorizer(max_features=5000)
        X = vectorizer.fit_transform(df['text']).toarray()
        y = df['sentiment']

        # Train SVM model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        # Overcoming Imbalanced Dataset with SMOTE
        smote = SMOTE(random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

        # Membuat model SVM
        model = SVC(kernel='linear')  # Anda bisa mengubah kernel sesuai kebutuhan

        # Melatih model
        model.fit(X_train_smote, y_train_smote)

        y_pred = model.predict(X_test)

        # Predict sentiment
        y_pred_all = model.predict(X)

        df['sentiment'] = pd.Series(y_pred_all).replace({2: "positive", 1: "neutral", 0: "negative"})

        positive_data = df[df['sentiment'] == 'positive']
        neutral_data = df[df['sentiment'] == 'neutral']
        negative_data = df[df['sentiment'] == 'negative']

        # Display sentiment results
        st.write("Sentiment Results:")

        # Pilih kelas sentimen untuk ditampilkan
        sentiment_class_option = st.selectbox("Select sentiment class to display", 
                                ('positive', 'neutral', 'negative'),
                                index=0)

        # Menentukan data yang dipilih berdasarkan sentiment_class
        if sentiment_class_option == 'positive':
            selected_data = positive_data
        elif sentiment_class_option == 'neutral':
            selected_data = neutral_data
        else:
            selected_data = negative_data

        # Tampilkan data yang dipilih
        st.write(f"Displaying {sentiment_class_option} sentiment data:")
        st.dataframe(selected_data[['text', 'sentiment']].head())

    # Topic Extraction
    st.write("### Topic Extraction")
    if 'sentiment' in df.columns and 'text' in df.columns:
        sentiment_classes = df['sentiment'].unique()
        contexts = {}

        for sentiment in sentiment_classes:
            posts = df[df['sentiment'] == sentiment]['text'].tolist()
            contexts[sentiment] = " ".join(posts)

        # User input for topic extraction
        sentiment_class = st.selectbox("Select sentiment class for topic extraction", 
                                        ('positive','neutral','negative'),
                                        index=0)
        if sentiment_class == 'positive':
            selected_data = positive_data
        elif sentiment_class == 'neutral':
            selected_data = neutral_data
        else:
            selected_data = negative_data

        if st.button("Extract Topics"):
            if sentiment_class in contexts:
                response = chat(contexts[sentiment_class], "", f"What are the main topics discussed in the {sentiment_class} class?")
                
                # Display the response from the LLM
                st.write("### Extracted Topics:")
                st.write(response["answer"])
            else:
                st.error("Selected sentiment class does not exist in the contexts.")
    else:
        st.error("The uploaded CSV file must contain 'sentiment' and 'text' columns.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is your question about sentiment analysis?"):
    # Get chat history if not Null
    messages_history = st.session_state.get("messages", [])
    history = "\n".join([f'{msg["role"]}: {msg["content"]}' for msg in messages_history]) or " "

    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get response from AI assistant
    response = chat("", history, prompt)
    answer = response["answer"]
    input_tokens = response["input_tokens"]
    completion_tokens = response["completion_tokens"]       

    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(answer)
        container = st.container(border=True)
        container.write(f"Input Tokens: {input_tokens}")
        container.write(f"Completion Tokens: {completion_tokens}")
    
    # Display chat history in an expander
    with st.expander("See Chat History"):
        st.write("*History Chat:*")
        st.code(history)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": answer})