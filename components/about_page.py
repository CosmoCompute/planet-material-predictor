import streamlit as st
import os
from components import local_def

# Load your custom CSS
local_def.load_css("assets/style.css")

def about_us():
    st.markdown("""
        <h2 class="about-section-header">
           👥 Meet Our Team     
        </h2>
    """, unsafe_allow_html=True)

    team_members = [
        {
            "name": "Arijit Chowdhury",
            "role": "Leader",
            "contribution": "Frontend->UI, Data Analysis, Data visualization, Backend-> logic-building, Model-training, Logic-building, Database-Management",
            "bio": "B.Sc Student at University of Calcutta in Computer Science",
            "github": "https://github.com/student-Arijit",
            "email": "arijitchowdhury4467@gmail.com" 
        },
        {
            "name": "Swarnabha Halder",
            "role": "Co-Leader",
            "contribution": "UI, Data Analysis, Model-training, Logic Building",
            "bio": "B.Tech in Computer Science and Engineering with Specialization in Data Science from Sikkim Manipal Institute of Technology ",
            "github": "https://github.com/swarnabha-dev",
            "email": "swarnabhahalder80137@gmail.com"
        }
    ]

    # Mix of GitHub avatar (URL) and local image path
    profile = [
        "https://avatars.githubusercontent.com/u/143516210?v=4",  # Remote image
        "assets/team/DPP40269.jpg"  # Local image
    ]

    st.markdown("""<br>""", unsafe_allow_html=True)

    for i, member in enumerate(team_members):
        try:
            image_src = profile[i]
        except IndexError:
            image_src = "https://via.placeholder.com/120"  # Safe fallback

        with st.container():
            col1, col2 = st.columns([1, 3])
            with col1:
                if image_src.startswith("http"):
                    # Use background-image for URL
                    st.markdown(f"""
                    <div style="
                        width: 120px;
                        height: 120px;
                        border-radius: 50%;
                        background-image: url('{image_src}');
                        background-size: cover;
                        background-position: center;
                        overflow: hidden;
                        border: 2px solid white;
                        margin-top: 10px;
                    ">
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    # Render local file using st.image, or fallback if missing
                    if os.path.exists(image_src):
                        st.image(image_src, width=120)
                    else:
                        st.image("https://via.placeholder.com/120", width=120)

            with col2:
                st.markdown(f"""
                <div class="team-card">
                    <h3 style="color: #FFFF; margin-bottom: 0.5rem;">{member['name']}</h3>
                    <h4 style="color: #FFFF; margin-bottom: 1rem;">{member['role']}</h4>
                    <p style="margin-bottom: 1rem;">{member['bio']}</p>
                    <div class="contact-info">
                        <strong>Contact:</strong><br>
                        {member['email']}<br>
                        <a href="{member['github']}" target="_blank">GitHub Profile</a>
                    </div>
                </div>
                """, unsafe_allow_html=True)
