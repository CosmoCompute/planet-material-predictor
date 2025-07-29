import streamlit as st
import os
import base64
from components import local_def

# Load your custom CSS
local_def.load_css("assets/style.css")

# Helper to base64-encode local image
def encode_image_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

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

    # Mixed source: URL + local image
    profile = [
        "https://avatars.githubusercontent.com/u/143516210?v=4",  # Remote
        "assets/team/10001314851.jpg"  # Local
    ]

    st.markdown("<br>", unsafe_allow_html=True)

    for i, member in enumerate(team_members):
        try:
            image_src = profile[i]
        except IndexError:
            image_src = "https://via.placeholder.com/120"

        with st.container():
            col1, col2 = st.columns([1, 3])
            with col1:
                # Render image with consistent class
                if image_src.startswith("http"):
                    st.markdown(f"""
                    <img src="{image_src}" class="profile-photo"/>
                    """, unsafe_allow_html=True)
                elif os.path.exists(image_src):
                    encoded = encode_image_to_base64(image_src)
                    st.markdown(f"""
                    <img src="data:image/jpeg;base64,{encoded}" class="profile-photo"/>
                    """, unsafe_allow_html=True)
                else:
                    st.image("https://via.placeholder.com/120", width=120)

            with col2:
                st.markdown(f"""
                <div class="team-card">
                    <h3>{member['name']}</h3>
                    <h4>{member['role']}</h4>
                    <p>{member['bio']}</p>
                    <div class="contact-info">
                        <strong>Contact:</strong><br>
                        {member['email']}<br>
                        <a href="{member['github']}" target="_blank">GitHub Profile</a>
                    </div>
                </div>
                """, unsafe_allow_html=True)
