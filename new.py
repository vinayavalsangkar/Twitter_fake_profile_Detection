import pandas as pd
import numpy as np
import streamlit as st
from joblib import load
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# Load the trained model
@st.cache_resource
def load_model():
    model = load('random_forest_model.joblib')
    return model


# Load the encoder
@st.cache_resource
def load_encoder():
    encoder = load('encoder.joblib')
    return encoder


# Define prediction function
def predict(model, input_data):
    prediction = model.predict(input_data)
    return prediction


# Function to load data for model analysiss
def load_data():
    # Load the real and fake user data

    real_users = pd.read_csv("C:\\Users\\VINAYA\\OneDrive\\Desktop\\TwitterProjectn\\TwitterProject\\realusers.csv")
    fake_users = pd.read_csv("C:\\Users\\VINAYA\\OneDrive\\Desktop\\TwitterProjectn\\TwitterProject\\fakeusers.csv")

    # Assign labels: 1 for real users and 0 for fake users
    real_users['label'] = 1
    fake_users['label'] = 0

    # Combine the datasets
    data = pd.concat([real_users, fake_users], ignore_index=True)

    # Preprocessing
    # Map 'lang' values to codes
    lang_list = list(enumerate(np.unique(data['lang'])))
    lang_dict = {name: i for i, name in lang_list}
    data['lang_code'] = data['lang'].map(lambda lang: lang_dict[lang]).astype(int)

    # Feature Engineering
    data['followers_friends_ratio'] = data['followers_count'] / (data['friends_count'] + 1)
    data['statuses_favourites_ratio'] = data['statuses_count'] / (data['favourites_count'] + 1)

    # Select features and target variable
    X = data[['statuses_count', 'followers_count', 'friends_count', 'favourites_count', 'listed_count', 'lang_code',
              'followers_friends_ratio', 'statuses_favourites_ratio']]
    y = data['label']

    # One-hot encode the lang_code column
    encoder = OneHotEncoder(drop='first')
    X_encoded = encoder.fit_transform(X[['lang_code']])
    X_encoded_df = pd.DataFrame(X_encoded.toarray(), columns=encoder.get_feature_names_out(['lang_code']))
    X = pd.concat([X.drop(columns=['lang_code']), X_encoded_df], axis=1)

    return X, y, encoder


# Main function to run the Streamlit app
def main():
    # Load the model and encoder
    model = load_model()
    encoder = load_encoder()

    # Load data for model analysis
    X, y, _ = load_data()

    # Split data for display purposes
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Title and description
    st.title('Twitter Fake Profile Detection')
    st.markdown("""
    <style>
    .reportview-container {
        background: #f5f5f5;
    }
    .sidebar .sidebar-content {
        background: #f5f5f5;
    }
    </style>
    """, unsafe_allow_html=True)

    # Tabs for different sections
    tab1, tab2 = st.tabs(["User Input", "Model Analysis"])

    with tab1:
        st.markdown("## User Input Features")

        # Sidebar for user input
        with st.sidebar:
            st.header("Input Features")

            statuses_count = st.number_input("Statuses Count", min_value=0)
            followers_count = st.number_input("Followers Count", min_value=0)
            friends_count = st.number_input("Friends Count", min_value=0)
            favourites_count = st.number_input("Favourites Count", min_value=0)
            listed_count = st.number_input("Listed Count", min_value=0)
            lang_code = st.number_input("Language Code", min_value=0)

        # Calculate additional features
        followers_friends_ratio = followers_count / (friends_count + 1)
        statuses_favourites_ratio = statuses_count / (favourites_count + 1)

        # Create DataFrame for input
        input_df = pd.DataFrame({
            'statuses_count': [statuses_count],
            'followers_count': [followers_count],
            'friends_count': [friends_count],
            'favourites_count': [favourites_count],
            'listed_count': [listed_count],
            'lang_code': [lang_code],
            'followers_friends_ratio': [followers_friends_ratio],
            'statuses_favourites_ratio': [statuses_favourites_ratio]
        })

        # One-hot encode lang_code
        lang_code_encoded = encoder.transform(input_df[['lang_code']])
        lang_code_encoded_df = pd.DataFrame(lang_code_encoded.toarray(),
                                            columns=encoder.get_feature_names_out(['lang_code']))
        input_df = pd.concat([input_df.drop(columns=['lang_code']), lang_code_encoded_df], axis=1)

        # Display input features
        st.markdown("### Input Data Summary")
        st.dataframe(input_df)

        # Visualization
        st.markdown("### Visualizations")
        col1, col2 = st.columns(2)

        with col1:
            st.bar_chart(input_df[['statuses_count', 'followers_count', 'friends_count']])

        with col2:
            st.bar_chart(input_df[['favourites_count', 'listed_count', 'followers_friends_ratio']])

        # Prediction
        if st.button("Detect Fake Profile"):
            prediction = predict(model, input_df)
            if prediction[0] == 1:
                st.markdown("<h1 style='font-size:24px; font-weight:bold; color:green;'>Prediction: Real Profile</h1>",
                            unsafe_allow_html=True)
            else:
                st.markdown("<h1 style='font-size:24px; font-weight:bold; color:red;'>Prediction: Fake Profile</h1>",
                            unsafe_allow_html=True)

    with tab2:
        st.markdown("## Model Analysis")

        # Display cross-validation scores
        cv_scores = cross_val_score(model, X, y, cv=3)
        st.write("Cross-Validation Scores:", cv_scores)
        st.write("Mean Cross-Validation Score:", cv_scores.mean())

        # Evaluate the model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.write("Accuracy:", accuracy)

        st.markdown("### Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())

        st.markdown("### Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        st.pyplot(fig)


# Run the app
if __name__ == '__main__':
    main()
