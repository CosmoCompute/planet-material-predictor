import streamlit as st
import base64
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Our Team", page_icon="👥", layout="wide")

# --- PATHS ---
# Update this path to where your CSS file is located
CSS_FILE = "assets/style.css" 
IMAGE_DIR = "assets/team"

# --- LOAD CSS ---
def load_css(file_path):
    with open(file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css(CSS_FILE)

# --- HELPER TO ENCODE IMAGES ---
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# --- TEAM DATA ---
team_members = [
    {
        "name": "Mike Cannon-Brookes",
        "role": "Co-Founder & Co-CEO",
        "image_path": os.path.join(IMAGE_DIR, "10001314851.jpg"), # Replace with your image paths
        "description": "Mike is a visionary leader driving our company's mission forward with passion and innovation.",
    },
    # {
    #     "name": "Scott Farquhar",
    #     "role": "Co-Founder & Co-CEO",
    #     "image_path": os.path.join(ASSETS_DIR, "scott.png"),
    #     "description": "Scott's strategic mindset and focus on culture have been instrumental in our growth and success.",
    # },
    # {
    #     "name": "Sri Viswanath",
    #     "role": "Chief Technology Officer",
    #     "image_path": os.path.join(ASSETS_DIR, "sri.png"),
    #     "description": "Sri leads our technology strategy, building robust and scalable platforms for the future.",
    # },
    # {
    #     "name": "Anu Bharadwaj",
    #     "role": "Head of Enterprise",
    #     "image_path": os.path.join(ASSETS_DIR, "anu.png"),
    #     "description": "Anu is dedicated to delivering exceptional value and solutions to our enterprise customers.",
    # },
    # {
    #     "name": "Erika Fisher",
    #     "role": "CAO & General Counsel",
    #     "image_path": os.path.join(ASSETS_DIR, "erika.png"),
    #     "description": "Erika oversees our administrative and legal functions, ensuring operational excellence.",
    # },
    # {
    #     "name": "James Beer",
    #     "role": "Chief Financial Officer",
    #     "image_path": os.path.join(ASSETS_DIR, "james.png"),
    #     "description": "James manages the financial health of the company, guiding our long-term economic strategy.",
    # },
]

# --- PAGE LAYOUT ---
st.markdown('<h1 class="main-title">Our leadership team</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">With over 100 years of combined experience, we\'ve got a well-seasoned team at the helm.</p>',
    unsafe_allow_html=True
)
st.write("---")

# --- GRID LAYOUT FOR TEAM MEMBERS ---
cols_per_row = 3
# Create a list of columns, e.g., [col1, col2, col3]
cols = st.columns(cols_per_row)

# Define a list of background colors, it will cycle through them
bg_colors = ["#f8dc7a", "#6de2c5", "#f5c3bd", "#c5e26d", "#6db6e2", "#e2a06d"]

# Loop through team members and assign them to columns
for i, member in enumerate(team_members):
    with cols[i % cols_per_row]:
        # Encode local image
        encoded_image = get_base64_image(member["image_path"])
        
        # Determine the background color from the cycle
        color = bg_colors[i % len(bg_colors)]
        
        st.markdown(f"""
        <div class="team-card">
            <div class="image-container" style="background-color: {color};">
                <img src="data:image/png;base64,{encoded_image}" class="profile-image">
                <div class="overlay">
                    <div class="description-text">{member['description']}</div>
                </div>
            </div>
            <h3>{member['name']}</h3>
            <p class="role">{member['role']}</p>
        </div>
        """, unsafe_allow_html=True)
        # Add some vertical space between rows
        st.markdown("<br>", unsafe_allow_html=True)




    # team_members = [
    #     {
    #         "name": "Arijit Chowdhury",
    #         "role": "Leader",
    #         "contribution": "Frontend->UI, Data Analysis, Data visualization, Backend-> logic-building, Model-training, Logic-building, Database-Management",
    #         "bio": "B.Sc Student at University of Calcutta in Computer Science",
    #         "github": "https://github.com/student-Arijit",
    #         "email": "arijitchowdhury4467@gmail.com" 
    #     },
    #     {
    #         "name": "Swarnabha Halder",
    #         "role": "Co-Leader",
    #         "contribution": "UI, Data Analysis, Model-training, Logic Building",
    #         "bio": "B.Tech in Computer Science and Engineering with Specialization in Data Science from Sikkim Manipal Institute of Technology ",
    #         "github": "https://github.com/swarnabha-dev",
    #         "email": "swarnabhahalder80137@gmail.com"
    #     }
    # ]

    # # Mixed source: URL + local image
    # profile = [
    #     "https://avatars.githubusercontent.com/u/143516210?v=4",  # Remote
    #     "assets/team/10001314851.jpg"  # Local
    # ]

   
