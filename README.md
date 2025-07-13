# IPMI Alumni Networking Bridge: AI-Powered Matchmaking App

## Overview
This Streamlit-based prototype app bridges diverse paths among IPMI International Business School alumni, fostering meaningful connections by uncovering shared insights, experiences, and opportunities. Leveraging a simple Retrieval-Augmented Generation (RAG)-inspired system via TF-IDF vectorization, it enables agentic AI matchmaking: Users input their professional, cultural, academic, and interest details to discover compatible alumni profiles from a database (currently dummy CSV data). This reveals synergies—such as linking an entrepreneur with innovation goals to a sustainability expert for eco-business collaborations—promoting interdisciplinary partnerships and global mentorships.

Ideal for hybrid networking events, the app enhances IPMI's alumni community of over 3,000 graduates worldwide, emphasizing inclusivity across varied backgrounds to build trust and lasting bonds.

## Features
- **Interactive Profile Input:** Alumni enter details to simulate real-time engagement.
- **AI-Driven Matching:** Retrieves top similar profiles using cosine similarity, highlighting potential for shared goals and collaborations.
- **Follow-Up Prompts:** Encourages scheduling virtual meets or joint projects post-matching.
- **Lightweight & Error-Free:** Built with minimal files for quick deployment on Streamlit's free tier; uses dummy data to avoid risks.

## Technologies
- Python 3.x
- Streamlit for UI
- Pandas for data handling
- Scikit-learn for TF-IDF and similarity computations

## Setup and Installation
1. **Clone the Repository:**
git clone https://github.com/yourusername/ipmi-alumni-networking-app.git
cd ipmi-alumni-networking-app


2. **Install Dependencies:**
Ensure Python is installed, then run:
pip install -r requirements.txt


3. **Run Locally:**
streamlit run app.py


Access at `http://localhost:8501` to test matchmaking.

## Deployment on Streamlit (Free Tier)
1. Push your repo to GitHub.
2. Visit [streamlit.io](https://streamlit.io), sign in, and create a new app from your repo.
3. Select `app.py` as the entry point—deploys in minutes with no costs for this prototype.

## Usage
- Launch the app and fill in your profile (e.g., Profession: "Entrepreneur", Interests: "Innovation, cultural exchanges").
- Click "Find Matches" to view suggested alumni, revealing shared threads for potential mentorships or projects.
- Expand later: Replace `dummy_alumni.csv` with real (anonymized) data for authentic connections.

## Data Format (dummy_alumni.csv)
A simple CSV with columns: `Name`, `Profession`, `Cultural Background`, `Academic Background`, `Interests`. Example row:
Dana Afriza,Finance Analyst,American,Harvard MBA,Global markets, innovation


## Contributing
Contributions are welcome to enhance features, like integrating advanced RAG with LLMs or expanding the database. Fork the repo, create a branch, and submit a pull request—let's collaborate to strengthen IPMI's global network!


## License
MIT License—free to use and adapt for fostering alumni synergies.

For questions or ideas on evolving this into a full networking platform, connect via GitHub issues. Together, we bridge diverse horizons into unified opportunities!
