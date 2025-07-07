import streamlit as st
from components import local_def

local_def.load_css("assets/style.css")

def about_us():
    st.markdown("""
        <h2 class="about-section-header">
           👥 Meet Our Team     
        </h2>
    """, unsafe_allow_html=True)

    team_members=[
        {
            "name": "Arijit Chowdhury",
            "role": "Leader",
            "contribution": "Frontend->UI, Data Analysis, Data visualization, Backend-> logic-building, Model-training, Logic-building, Database-Management",
            "bio": "B.Sc Student at University of Calcutta in Computer Science",
            "github": "https://github.com/student-Arijit",
            "email": "arijitchowdhury4467@gmail.com" 
        }
    ]

    profile=[
        "https://avatars.githubusercontent.com/u/143516210?v=4"
    ]

    st.markdown("""<br>""", unsafe_allow_html=True)

    for i, member in enumerate(team_members):
        with st.container():
            col1, col2 = st.columns([1, 3])
            with col1:
                st.markdown(f"""
                <div style="width: 120px;
                height: 120px;
                border-radius: 50%;
                background-image: url('{profile[i]}');
                background-size: cover;
                background-position: center;
                overflow: hidden;">
                </div>
                """, unsafe_allow_html=True)
        
            with col2:
                st.markdown(f"""
                <div class="team-card">
                    <h3 style="color: #2E86AB; margin-bottom: 0.5rem;">{member['name']}</h3>
                    <h4 style="color: #666; margin-bottom: 1rem;">{member['role']}</h4>
                    <p style="margin-bottom: 1rem;">{member['bio']}</p>
                    <div class="contact-info">
                        <strong>Contact:</strong><br>
                        {member['email']}<br>
                        <a href="{member['github']}" target="_blank">GitHub Profile</a>
                    </div>
                </div>
                """, unsafe_allow_html=True)