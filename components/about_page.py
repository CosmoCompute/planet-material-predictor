import streamlit as st
import os
import base64
# Assuming 'local_def' is in 'components' folder
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
            <h2>Our leadership team</h2>
            <p>With over 100 years of combined experience, we've got a well-seasoned team at the helm.</p>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # --- TEAM DATA (No changes here) ---
    team_members = [
        {
            "name": "Swarnabha Halder",
            "role": "CO-LEADER",
            "description": "B.Tech in Computer Science and Engineering with Specialization in Data Science from Sikkim Manipal Institute of Technolog.<br>contribution: UI, Data Analysis, Model-training, Logic Building.<br>github: https://github.com/swarnabha-dev.<br>email: h.swarnabha@gmail.com",
            "image_path": "assets/team/10001314851.jpg",
            "bg_color": "#fdbb2d" # Yellow
        },
        {
            "name": "Arijit Chowdhury",
            "role": "Leader",
            "description": "B.Sc Student at University of Calcutta in Computer Science.<br>contribution: Frontend->UI, Data Analysis, Data visualization, Backend-> logic-building, Model-training, Logic-building, Database-Management.<br>github: https://github.com/student-Arijit.<br>email: arijitchowdhury4467@gmail.com",
            "image_path": "https://avatars.githubusercontent.com/u/143516210?v=4",
            "bg_color": "#1fe6a8" # Green
        }
        # {
        #     "name": "Sri Viswanath",
        #     "role": "CHIEF TECHNOLOGY OFFICER",
        #     "description": "Your short description for Sri. This text appears on hover.",
        #     "image_path": "assets/team/sri.jpg",
        #     "bg_color": "#f6a7a6" # Pink
        # },
        # {
        #     "name": "Anu Bharadwaj",
        #     "role": "HEAD OF ENTERPRISE & CLOUD PLATFORM",
        #     "description": "Your short description for Anu. This text appears on hover.",
        #     "image_path": "assets/team/anu.jpg",
        #     "bg_color": "#1fe6a8" # Green
        # },
        # {
        #     "name": "Erika Fisher",
        #     "role": "CHIEF ADMINISTRATIVE OFFICER",
        #     "description": "Your short description for Erika. This text appears on hover.",
        #     "image_path": "assets/team/erika.jpg",
        #     "bg_color": "#2873f4" # Blue
        # },
        # {
        #     "name": "James Beer",
        #     "role": "CHIEF FINANCIAL OFFICER",
        #     "description": "Your short description for James. This text appears on hover.",
        #     "image_path": "assets/team/james.jpg",
        #     "bg_color": "#fdbb2d" # Yellow
        # }
    ]

    # --- RENDER TEAM MEMBERS IN A GRID ---
    cols = st.columns(3)

    for i, member in enumerate(team_members):
        with cols[i % 3]:
            image_src = ""
            if member["image_path"].startswith("http"):
                image_src = member["image_path"]
            elif os.path.exists(member["image_path"]):
                encoded = encode_image_to_base64(member["image_path"])
                image_src = f"data:image/jpeg;base64,{encoded}"
            else:
                image_src = "https://via.placeholder.com/300"
            
            # --- MODIFIED HTML STRUCTURE ---
            # The overlay div is removed and a new .member-description div is added.
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
                    <p>{member['description']}</p>
                </div>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)
