import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import requests
import json
from typing import List, Dict, Optional

# Page configuration
st.set_page_config(
    page_title="IPMI Alumni Networking Bridge",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        color: white;
        text-align: center;
        margin: 0;
    }
    .match-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #2a5298;
        margin: 1rem 0;
    }
    .similarity-score {
        background: #e3f2fd;
        padding: 0.5rem;
        border-radius: 5px;
        display: inline-block;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_alumni_data():
    """Load alumni data with error handling"""
    try:
        # Try multiple possible paths
        possible_paths = [
            'data/dummy_alumni.csv',
            'dummy_alumni.csv',
            './data/dummy_alumni.csv'
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                df = pd.read_csv(path)
                return df
        
        # If no file found, create dummy data
        st.warning("Alumni data file not found. Using fallback data.")
        return create_dummy_data()
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return create_dummy_data()

def create_dummy_data():
    """Create fallback dummy data if CSV file is not found"""
    data = {
        'Name': [
            'Dana Afriza',
            'Siddartha Simpson', 
            'Elviera Ramirez',
            'Bung Towel',
            'Kun Welly Never Surrender',
            'Maria Santos',
            'Ahmad Rahman',
            'Jennifer Chen',
            'Roberto Silva',
            'Priya Patel'
        ],
        'Profession': [
            'Finance Analyst',
            'Entrepreneur',
            'Tech Developer',
            'Marketing Specialist',
            'Sustainability Expert',
            'Product Manager',
            'Data Scientist',
            'Business Consultant',
            'Operations Manager',
            'Digital Marketing'
        ],
        'Cultural Background': [
            'American',
            'Indonesian',
            'Latin American',
            'Indian',
            'Chinese',
            'Brazilian',
            'Malaysian',
            'Taiwanese',
            'Mexican',
            'Indian'
        ],
        'Academic Background': [
            'Harvard MBA',
            'Undergraduate Business',
            'Computer Science',
            'Marketing',
            'Environmental Studies',
            'Engineering MBA',
            'Statistics PhD',
            'Business Administration',
            'Operations Research',
            'Digital Media'
        ],
        'Interests': [
            'Global markets, innovation',
            'Startup ecosystems, cultural exchanges',
            'AI applications, coding',
            'Brand strategy, diversity',
            'Eco-business, collaborations',
            'Product development, user experience',
            'Machine learning, analytics',
            'Strategy consulting, leadership',
            'Supply chain, efficiency',
            'Social media, content creation'
        ]
    }
    return pd.DataFrame(data)

def preprocess_data(df):
    """Preprocess the alumni data for matching"""
    # Combine features for vectorization
    df['Profile'] = (
        df['Profession'] + ' ' + 
        df['Cultural Background'] + ' ' + 
        df['Academic Background'] + ' ' + 
        df['Interests']
    )
    return df

def get_ai_insights(user_profile: str, matches: pd.DataFrame) -> str:
    """Get AI-powered insights about the matches"""
    
    # Check if API key is available (from environment or secrets)
    api_key = None
    try:
        # Try to get from Streamlit secrets
        if 'openrouter' in st.secrets:
            api_key = st.secrets['openrouter']['api_key']
    except:
        pass
    
    if not api_key:
        try:
            # Try to get from environment variables
            api_key = os.environ.get('OPENROUTER_API_KEY')
        except:
            pass
    
    if not api_key:
        return generate_fallback_insights(user_profile, matches)
    
    try:
        # Prepare the prompt for AI analysis
        matches_text = ""
        for _, match in matches.iterrows():
            matches_text += f"- {match['Name']}: {match['Profession']}, {match['Interests']}\n"
        
        prompt = f"""
        Analyze this networking scenario:
        
        User Profile: {user_profile}
        
        Potential Matches:
        {matches_text}
        
        Provide brief, actionable insights about:
        1. Why these matches work well
        2. Specific collaboration opportunities
        3. Networking strategy suggestions
        
        Keep response under 200 words and focus on practical advice.
        """
        
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "qwen/qwq-32b:free",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 300,
                "temperature": 0.7
            },
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            return generate_fallback_insights(user_profile, matches)
            
    except Exception as e:
        st.warning(f"AI insights temporarily unavailable: {e}")
        return generate_fallback_insights(user_profile, matches)

def generate_fallback_insights(user_profile: str, matches: pd.DataFrame) -> str:
    """Generate basic insights when AI is not available"""
    insights = [
        "🎯 **Networking Strategy**: Focus on shared interests and complementary skills.",
        "💡 **Collaboration Ideas**: Look for cross-industry opportunities and knowledge exchange.",
        "🌍 **Cultural Bridge**: Leverage diverse backgrounds for global perspectives.",
        "📈 **Growth Potential**: Consider mentorship opportunities and skill development.",
        "🤝 **Next Steps**: Reach out with specific project ideas or collaboration proposals."
    ]
    
    return "\n\n".join(insights)

def find_matches(user_profile: str, df: pd.DataFrame, top_n: int = 3):
    """Find similar alumni profiles using TF-IDF and cosine similarity"""
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=1000,
        ngram_range=(1, 2)
    )
    
    # Fit and transform the alumni profiles
    tfidf_matrix = vectorizer.fit_transform(df['Profile'])
    
    # Transform user profile
    user_vector = vectorizer.transform([user_profile])
    
    # Calculate cosine similarities
    similarities = cosine_similarity(user_vector, tfidf_matrix).flatten()
    
    # Get top matches
    top_indices = np.argsort(similarities)[-top_n:][::-1]
    
    # Return matches with similarity scores
    matches = df.iloc[top_indices].copy()
    matches['Similarity'] = similarities[top_indices]
    
    return matches

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🌐 IPMI Alumni Networking Bridge</h1>
        <p style="color: white; text-align: center; margin: 0;">
            AI-Powered Matchmaking for Meaningful Professional Connections
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Your Profile")
        st.write("Fill in your details to discover compatible alumni connections.")
        
        # API Key input (optional)
        with st.expander("🔧 Advanced Settings"):
            st.write("For enhanced AI insights, you can optionally provide an OpenRouter API key:")
            api_key_input = st.text_input(
                "OpenRouter API Key (Optional)", 
                type="password",
                help="Get your free API key from openrouter.ai"
            )
            if api_key_input:
                os.environ['OPENROUTER_API_KEY'] = api_key_input
    
    # Load data
    df = load_alumni_data()
    df = preprocess_data(df)
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("👤 Your Information")
        
        profession = st.text_input(
            "Profession",
            placeholder="e.g., Data Scientist, Entrepreneur, Consultant",
            help="Your current professional role or area of expertise"
        )
        
        cultural_bg = st.text_input(
            "Cultural Background",
            placeholder="e.g., Indonesian, American, Multi-cultural",
            help="Your cultural or geographical background"
        )
        
        academic_bg = st.text_input(
            "Academic Background", 
            placeholder="e.g., MBA, Computer Science, Business Administration",
            help="Your educational background or area of study"
        )
        
        interests = st.text_area(
            "Interests & Goals",
            placeholder="e.g., Innovation, sustainability, startup ecosystems, mentoring",
            help="Your professional interests, goals, and areas you'd like to explore",
            height=100
        )
    
    with col2:
        st.subheader("🎯 Matching Preferences")
        
        num_matches = st.slider("Number of matches to show", 1, 5, 3)
        
        match_criteria = st.multiselect(
            "Focus on matching by:",
            ["Professional expertise", "Cultural background", "Academic field", "Interests & goals"],
            default=["Professional expertise", "Interests & goals"]
        )
        
        st.info("💡 Tip: Be specific about your interests and goals for better matches!")
    
    # Matching button and results
    if st.button("🔍 Find My Matches", type="primary"):
        if profession and cultural_bg and academic_bg and interests:
            
            # Create user profile
            user_profile = f"{profession} {cultural_bg} {academic_bg} {interests}"
            
            with st.spinner("Finding your perfect matches..."):
                # Find matches
                matches = find_matches(user_profile, df, num_matches)
                
                if len(matches) > 0:
                    st.success(f"Found {len(matches)} great matches for you!")
                    
                    # Display matches
                    st.subheader("🤝 Your Alumni Matches")
                    
                    for idx, (_, match) in enumerate(matches.iterrows(), 1):
                        similarity_pct = match['Similarity'] * 100
                        
                        st.markdown(f"""
                        <div class="match-card">
                            <h4>#{idx} {match['Name']}</h4>
                            <p><strong>🏢 Profession:</strong> {match['Profession']}</p>
                            <p><strong>🌍 Cultural Background:</strong> {match['Cultural Background']}</p>
                            <p><strong>🎓 Academic Background:</strong> {match['Academic Background']}</p>
                            <p><strong>💡 Interests:</strong> {match['Interests']}</p>
                            <div class="similarity-score">
                                <strong>Match Score: {similarity_pct:.1f}%</strong>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # AI Insights
                    st.subheader("🧠 AI-Powered Networking Insights")
                    with st.spinner("Generating personalized insights..."):
                        insights = get_ai_insights(user_profile, matches)
                        st.markdown(insights)
                    
                    # Next steps
                    st.subheader("📞 Next Steps")
                    st.markdown("""
                    **Ready to connect?** Here are some ways to reach out:
                    
                    1. **📧 Send a personalized message** mentioning shared interests
                    2. **☕ Suggest a virtual coffee chat** to explore collaboration opportunities  
                    3. **🤝 Propose a specific project** where you could work together
                    4. **🎯 Join or create networking events** around common interests
                    
                    *Remember: The best networking conversations start with genuine curiosity about the other person's work and experiences.*
                    """)
                    
                else:
                    st.warning("No matches found. Try adjusting your profile information.")
        else:
            st.error("⚠️ Please fill in all fields to enable matchmaking.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>🎓 IPMI Alumni Networking Bridge | Connecting 3,000+ Alumni Worldwide</p>
        <p>Built with ❤️ for meaningful professional connections</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
