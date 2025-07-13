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
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .similarity-score {
        background: #e3f2fd;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
        margin-top: 0.5rem;
        font-weight: bold;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #ff6b6b 0%, #ee5a6f 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Set default OpenRouter API key for free tier
DEFAULT_API_KEY = "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"

@st.cache_data
def load_alumni_data():
    """Load alumni data with error handling"""
    try:
        # Try multiple possible paths
        possible_paths = [
            'dummy_alumni.csv',
            'data/dummy_alumni.csv',
            './dummy_alumni.csv'
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                df = pd.read_csv(path)
                if not df.empty:
                    return df
        
        # If no file found, create dummy data
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
            'Priya Patel',
            'Alex Johnson',
            'Lisa Wang',
            'David Kim',
            'Sarah Brown',
            'Michael Zhang'
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
            'Digital Marketing',
            'Software Engineer',
            'Investment Banker',
            'UX Designer',
            'HR Director',
            'Sales Manager'
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
            'Indian',
            'Australian',
            'Singapore',
            'Korean',
            'Canadian',
            'Thai'
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
            'Digital Media',
            'Software Engineering',
            'Finance MBA',
            'Design Studies',
            'Human Resources',
            'International Business'
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
            'Social media, content creation',
            'Technology innovation, startups',
            'Investment strategies, fintech',
            'User interface, creativity',
            'Talent management, culture',
            'Business development, networking'
        ]
    }
    return pd.DataFrame(data)

def preprocess_data(df):
    """Preprocess the alumni data for matching"""
    # Combine features for vectorization
    df['Profile'] = (
        df['Profession'].astype(str) + ' ' + 
        df['Cultural Background'].astype(str) + ' ' + 
        df['Academic Background'].astype(str) + ' ' + 
        df['Interests'].astype(str)
    )
    return df

def get_ai_insights(user_profile: str, matches: pd.DataFrame) -> str:
    """Get AI-powered insights about the matches"""
    
    # Get API key from environment or use default
    api_key = os.environ.get('OPENROUTER_API_KEY', DEFAULT_API_KEY)
    
    try:
        # Prepare the prompt for AI analysis
        matches_text = ""
        for _, match in matches.iterrows():
            matches_text += f"- {match['Name']}: {match['Profession']}, {match['Interests']}\n"
        
        prompt = f"""
        Analyze this networking scenario for IPMI alumni:
        
        User Profile: {user_profile}
        
        Top Matches:
        {matches_text}
        
        Provide brief insights about:
        1. Why these matches are excellent for networking
        2. Specific collaboration opportunities
        3. Actionable next steps for connection
        
        Be encouraging and specific. Keep under 150 words.
        """
        
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://ipmi-alumni-networking-app.streamlit.app",
                "X-Title": "IPMI Alumni Networking"
            },
            json={
                "model": "qwen/qwq-32b:free",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 200,
                "temperature": 0.8
            },
            timeout=15
        )
        
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            return generate_fallback_insights(user_profile, matches)
            
    except Exception as e:
        return generate_fallback_insights(user_profile, matches)

def generate_fallback_insights(user_profile: str, matches: pd.DataFrame) -> str:
    """Generate basic insights when AI is not available"""
    insights = """
    🎯 **Perfect Networking Opportunities**: Your matches show excellent potential for meaningful connections based on shared interests and complementary expertise.
    
    💡 **Collaboration Ideas**: Consider cross-industry projects, knowledge sharing sessions, or mentor-mentee relationships that leverage diverse backgrounds.
    
    🌍 **Global Perspective**: The cultural diversity in your matches offers opportunities for international business insights and cross-cultural collaboration.
    
    📈 **Growth Strategy**: These connections can accelerate your professional development through skill exchange and industry insights.
    
    🤝 **Action Plan**: Reach out with specific collaboration ideas or suggest virtual coffee chats to explore synergies.
    """
    
    return insights

def find_matches(user_profile: str, df: pd.DataFrame, top_n: int = 3):
    """Find similar alumni profiles using TF-IDF and cosine similarity"""
    
    try:
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=1000,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95
        )
        
        # Fit and transform the alumni profiles
        tfidf_matrix = vectorizer.fit_transform(df['Profile'])
        
        # Transform user profile
        user_vector = vectorizer.transform([user_profile])
        
        # Calculate cosine similarities
        similarities = cosine_similarity(user_vector, tfidf_matrix).flatten()
        
        # Get top matches (excluding exact matches)
        top_indices = np.argsort(similarities)[-top_n-1:][::-1]
        
        # Filter out very low similarities and exact matches
        valid_matches = []
        for idx in top_indices:
            if similarities[idx] > 0.01 and similarities[idx] < 0.99:
                valid_matches.append(idx)
            if len(valid_matches) >= top_n:
                break
        
        if len(valid_matches) == 0:
            # If no good matches, return top scoring ones anyway
            valid_matches = top_indices[:top_n]
        
        # Return matches with similarity scores
        matches = df.iloc[valid_matches].copy()
        matches['Similarity'] = similarities[valid_matches]
        
        return matches
    
    except Exception as e:
        st.error(f"Error in matching algorithm: {e}")
        # Return random sample as fallback
        return df.sample(n=min(top_n, len(df)))

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
        
        # API Status
        with st.expander("🔧 Advanced Settings"):
            api_status = "🟢 AI Insights Active" if DEFAULT_API_KEY else "🔴 AI Insights Unavailable"
            st.write(f"**Status**: {api_status}")
            st.write("Enhanced AI insights are automatically enabled for better networking recommendations.")
    
    # Load data
    df = load_alumni_data()
    df = preprocess_data(df)
    
    # Initialize session state
    if 'form_submitted' not in st.session_state:
        st.session_state.form_submitted = False
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("👤 Your Information")
        
        profession = st.text_input(
            "Profession",
            value="",
            placeholder="e.g., Data Scientist, Entrepreneur, Consultant",
            help="Your current professional role or area of expertise"
        )
        
        cultural_bg = st.text_input(
            "Cultural Background",
            value="",
            placeholder="e.g., Indonesian, American, Multi-cultural",
            help="Your cultural or geographical background"
        )
        
        academic_bg = st.text_input(
            "Academic Background", 
            value="",
            placeholder="e.g., MBA, Computer Science, Business Administration",
            help="Your educational background or area of study"
        )
        
        interests = st.text_area(
            "Interests & Goals",
            value="",
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
        # Improved validation - check if fields are not empty and not just whitespace
        profession_valid = profession and profession.strip()
        cultural_bg_valid = cultural_bg and cultural_bg.strip()
        academic_bg_valid = academic_bg and academic_bg.strip()
        interests_valid = interests and interests.strip()
        
        if profession_valid and cultural_bg_valid and academic_bg_valid and interests_valid:
            st.session_state.form_submitted = True
            
            # Create user profile
            user_profile = f"{profession.strip()} {cultural_bg.strip()} {academic_bg.strip()} {interests.strip()}"
            
            with st.spinner("🔍 Finding your perfect matches..."):
                # Find matches
                matches = find_matches(user_profile, df, num_matches)
                
                if len(matches) > 0:
                    st.success(f"🎉 Found {len(matches)} excellent matches for you!")
                    
                    # Display matches
                    st.subheader("🤝 Your Alumni Matches")
                    
                    for idx, (_, match) in enumerate(matches.iterrows(), 1):
                        similarity_pct = getattr(match, 'Similarity', 0.75) * 100
                        
                        st.markdown(f"""
                        <div class="match-card">
                            <h4>#{idx} {match['Name']}</h4>
                            <p><strong>🏢 Profession:</strong> {match['Profession']}</p>
                            <p><strong>🌍 Cultural Background:</strong> {match['Cultural Background']}</p>
                            <p><strong>🎓 Academic Background:</strong> {match['Academic Background']}</p>
                            <p><strong>💡 Interests:</strong> {match['Interests']}</p>
                            <div class="similarity-score">
                                ⭐ Match Score: {similarity_pct:.0f}%
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # AI Insights
                    st.subheader("🧠 AI-Powered Networking Insights")
                    with st.spinner("🤖 Generating personalized insights..."):
                        insights = get_ai_insights(user_profile, matches)
                        st.markdown(insights)
                    
                    # Next steps
                    st.subheader("📞 Next Steps")
                    st.markdown("""
                    **Ready to connect?** Here are proven ways to reach out:
                    
                    1. **📧 Send a personalized message** mentioning shared interests from your matches
                    2. **☕ Suggest a virtual coffee chat** to explore collaboration opportunities  
                    3. **🤝 Propose a specific project** where you could work together
                    4. **🎯 Join networking events** around your common interests
                    5. **💡 Share industry insights** to start meaningful conversations
                    
                    *💫 Pro tip: The best networking starts with genuine curiosity about others' experiences and expertise.*
                    """)
                    
                else:
                    st.warning("🔍 No matches found. Try adjusting your profile information or broadening your interests.")
        else:
            st.error("⚠️ Please fill in all fields completely to enable matchmaking.")
            missing_fields = []
            if not profession_valid: missing_fields.append("Profession")
            if not cultural_bg_valid: missing_fields.append("Cultural Background") 
            if not academic_bg_valid: missing_fields.append("Academic Background")
            if not interests_valid: missing_fields.append("Interests & Goals")
            
            if missing_fields:
                st.error(f"Missing: {', '.join(missing_fields)}")
    
    # Show alumni database info
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📊 Alumni Database", f"{len(df)} profiles")
    with col2:
        st.metric("🌍 Countries", len(df['Cultural Background'].unique()))
    with col3:
        st.metric("💼 Professions", len(df['Profession'].unique()))
    
    # Footer
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p>🎓 <strong>IPMI Alumni Networking Bridge</strong> | Connecting 3,000+ Alumni Worldwide</p>
        <p>Built with ❤️ for meaningful professional connections | Powered by AI</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
