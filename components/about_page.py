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
    github_icon = """
<a href="{github_url}" class="github-icon" target="_blank">
  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" fill="#181717" viewBox="0 0 16 16">
    <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 
           2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 
           0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94
           -.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53
           .63-.01 1.08.58 1.23.82.72 1.21 1.87.87 
           2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95
           0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 
           0 0 .67-.21 2.2.82.64-.18 1.32-.27 
           2-.27s1.36.09 2 .27c1.53-1.04 2.2-.82 
           2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 
           1.28.82 2.15 0 3.07-1.87 3.75-3.65 
           3.95.29.25.54.73.54 1.48 0 1.07-.01 
           1.93-.01 2.2 0 .21-.15.46-.55.38A8.01 
           8.01 0 0 1 0 8z"/>
  </svg>
</a>
"""



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
                        <a href="{member['github']}" target="_blank" class="github-icon">{github_icon.format(github_url=member['github'])}</a>
                        <a href="mailto:{member['email']}">{email_icon}</a>
                    </div>
                </div>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)
