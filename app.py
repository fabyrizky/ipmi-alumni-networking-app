import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dummy alumni data
df = pd.read_csv('dummy_alumni.csv')

# Combine features for vectorization (RAG-like retrieval prep)
df['Profile'] = df['Profession'] + ' ' + df['Cultural Background'] + ' ' + df['Academic Background'] + ' ' + df['Interests']

# TF-IDF Vectorizer for similarity matching
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['Profile'])

# Streamlit UI for interactive profile input
st.title('IPMI Alumni Networking Bridge')
st.write('Enter your profile details to discover matching alumni, uncovering shared insights and collaboration opportunities across diverse backgrounds.')

profession = st.text_input('Profession (e.g., Entrepreneur)')
cultural_bg = st.text_input('Cultural Background (e.g., Indonesian)')
academic_bg = st.text_input('Academic Background (e.g., MBA)')
interests = st.text_area('Interests (e.g., Innovation, cultural exchanges)')

if st.button('Find Matches'):
    if profession and cultural_bg and academic_bg and interests:
        # Build user profile string
        user_profile = profession + ' ' + cultural_bg + ' ' + academic_bg + ' ' + interests
        
        # Vectorize user input
        user_vector = vectorizer.transform([user_profile])
        
        # Compute cosine similarities (agentic matching)
        similarities = cosine_similarity(user_vector, tfidf_matrix)
        
        # Get top 3 matches
        top_indices = similarities[0].argsort()[-3:][::-1]
        matches = df.iloc[top_indices]
        
        st.write('### Suggested Alumni Matches:')
        st.write('These connections highlight potential synergies for mentorships, joint projects, or shared goal pursuits.')
        
        for i, row in matches.iterrows():
            st.write(f"**{row['Name']}**")
            st.write(f"- Profession: {row['Profession']}")
            st.write(f"- Cultural Background: {row['Cultural Background']}")
            st.write(f"- Academic Background: {row['Academic Background']}")
            st.write(f"- Interests: {row['Interests']}")
            st.write('---')  # Separator for clarity
        
        st.write('Follow up by reaching out—perhaps schedule a virtual coffee to explore interdisciplinary ideas!')
    else:
        st.write('Please fill in all fields to enable matchmaking.')
