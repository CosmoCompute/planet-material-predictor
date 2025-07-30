import streamlit as st
import os
import base64
from components import local_def

# Load your custom CSS
local_def.load_css("assets/style.css")

# Helper to base64-encode local image (keep as is)
def encode_image_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def about_us():
    # --- PAGE HEADER (No changes here) ---
    st.markdown("""
        <div style="text-align: center;">
            <h2>Our Cosmosompute Team</h2>
        
        </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    # --- SVG ICONS (GitHub icon is updated) ---
    github_icon = '<span class="github-icon"></span>'


    email_icon = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="currentColor"><path d="M22 6c0-1.1-.9-2-2-2H4c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V6zm-2 0-8 5-8-5h16zm0 12H4V8l8 5 8-5v10z"></path></svg>"""

    # --- TEAM DATA with new structure ---
    team_members = [
        {
            "name": "Arijit Chowdhury",
            "role": "LEADER",
            "bio": "A B.Sc Student at University of Calcutta, specializing in Computer Science.",
            "contribution": "Frontend UI, Data Analysis, Visualization, Backend Logic, Model Training, & Database Management.",
            "github": "https://github.com/student-Arijit",
            "email": "arijitchowdhury4467@gmail.com",
            "image_path": "https://avatars.githubusercontent.com/u/143516210?v=4",
            "bg_color": "#fdbb2d"
        },
        {
            "name": "Swarnabha Halder",
            "role": "CO-LEADER",
            "bio": "A B.Tech in Computer Science with a specialization in Data Science from SMIT.",
            "contribution": "UI Design, Data Analysis, Model Training, & Core Logic Building.",
            "github": "https://github.com/swarnabha-dev",
            "email": "swarnabhahalder80137@gmail.com",
            "image_path": "assets/team/10001314851.jpg", # 👈 IMPORTANT: Update path
            "bg_color": "#1fe6a8"
        },
        # Add other team members here following the same dictionary structure
        # {
        #     "name": "Swarnabha Halder",
        #     "role": "CO-LEADER",
        #     "bio": "A B.Tech in Computer Science with a specialization in Data Science from SMIT.",
        #     "contribution": "UI Design, Data Analysis, Model Training, & Core Logic Building.",
        #     "github": "https://github.com/swarnabha-dev",
        #     "email": "swarnabhahalder80137@gmail.com",
        #     "image_path": "assets/team/swarnabha.jpg", # 👈 IMPORTANT: Update path
        #     "bg_color": "#1fe6a8"
        # },
        # {
        #     "name": "Swarnabha Halder",
        #     "role": "CO-LEADER",
        #     "bio": "A B.Tech in Computer Science with a specialization in Data Science from SMIT.",
        #     "contribution": "UI Design, Data Analysis, Model Training, & Core Logic Building.",
        #     "github": "https://github.com/swarnabha-dev",
        #     "email": "swarnabhahalder80137@gmail.com",
        #     "image_path": "assets/team/swarnabha.jpg", # 👈 IMPORTANT: Update path
        #     "bg_color": "#1fe6a8"
        # },
        # {
        #     "name": "Swarnabha Halder",
        #     "role": "CO-LEADER",
        #     "bio": "A B.Tech in Computer Science with a specialization in Data Science from SMIT.",
        #     "contribution": "UI Design, Data Analysis, Model Training, & Core Logic Building.",
        #     "github": "https://github.com/swarnabha-dev",
        #     "email": "swarnabhahalder80137@gmail.com",
        #     "image_path": "assets/team/swarnabha.jpg", # 👈 IMPORTANT: Update path
        #     "bg_color": "#1fe6a8"
        # },
        # {
        #     "name": "Swarnabha Halder",
        #     "role": "CO-LEADER",
        #     "bio": "A B.Tech in Computer Science with a specialization in Data Science from SMIT.",
        #     "contribution": "UI Design, Data Analysis, Model Training, & Core Logic Building.",
        #     "github": "https://github.com/swarnabha-dev",
        #     "email": "swarnabhahalder80137@gmail.com",
        #     "image_path": "assets/team/swarnabha.jpg", # 👈 IMPORTANT: Update path
        #     "bg_color": "#1fe6a8"
        # },
    ]

    # --- RENDER TEAM MEMBERS IN A GRID ---
    cols = st.columns(len(team_members) if len(team_members) <= 3 else 3)
    for i, member in enumerate(team_members):
        with cols[i % 3]:
            image_src = "" # Handle image source logic
            if member["image_path"].startswith("http"):
                image_src = member["image_path"]
            elif os.path.exists(member["image_path"]):
                encoded = encode_image_to_base64(member["image_path"])
                image_src = f"data:image/jpeg;base64,{encoded}"
            else: image_src = "https://via.placeholder.com/300"
            
            # --- HTML STRUCTURE (No changes here) ---
            card_html = f"""
            <div class="team-member-card">
                <div class="image-container" style="background-color: {member['bg_color']};">
                    <img src="{image_src}" class="profile-img">
                </div>
                <div class="member-info">
                    <h3>{member['name']}</h3>
                    <p>{member['role']}</p>
                </div>
                <div class="member-description">
                    <div class="description-content">
                        <p><strong>Bio:</strong> {member['bio']}</p>
                        <p><strong>Contribution:</strong> {member['contribution']}</p>
                    </div>
                    <div class="social-links">
                        <a href="{member['github']}" target="_blank">{github_icon}</a>
                        <a href="mailto:{member['email']}">{email_icon}</a>
                    </div>
                </div>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)
