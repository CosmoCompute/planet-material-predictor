import streamlit as st
from PIL import Image
import requests
from io import BytesIO

# Page configuration
st.set_page_config(
    page_title="Smart Material Predictor - About",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E86AB;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #2E86AB, #A23B72);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .section-header {
        color: #2E86AB;
        font-size: 2rem;
        font-weight: bold;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 3px solid #2E86AB;
        padding-bottom: 0.5rem;
    }
    
    .team-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        border-left: 4px solid #2E86AB;
    }
    
    .company-info {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
        text-align: center;
    }
    
    .project-stats {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .tech-badge {
        background: #2E86AB;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        margin: 0.25rem;
        display: inline-block;
    }
    
    .contact-info {
        background: #f1f3f4;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🔬 Smart Material Predictor</h1>', unsafe_allow_html=True)

# Project Overview
st.markdown('<h2 class="section-header">Project Overview</h2>', unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    **Smart Material Predictor** is an advanced AI-powered platform that revolutionizes material science research 
    by predicting material properties using cutting-edge machine learning algorithms. Our system enables researchers 
    and engineers to accelerate material discovery, optimize compositions, and predict performance characteristics 
    with unprecedented accuracy.
    
    ### Key Features:
    - **AI-Powered Predictions**: Advanced neural networks for material property prediction
    - **Interactive Visualization**: Real-time data analysis and visualization tools
    - **Comprehensive Database**: Extensive material property database
    - **User-Friendly Interface**: Intuitive design for researchers of all levels
    - **Export Capabilities**: Generate detailed reports and export results
    """)

with col2:
    st.markdown("""
    <div class="project-stats">
        <h3>📊 Project Statistics</h3>
        <ul>
            <li><strong>Materials Analyzed:</strong> 10,000+</li>
            <li><strong>Prediction Accuracy:</strong> 95.2%</li>
            <li><strong>Active Users:</strong> 500+</li>
            <li><strong>Research Papers:</strong> 25+</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Technology Stack
st.markdown('<h2 class="section-header">Technology Stack</h2>', unsafe_allow_html=True)

tech_col1, tech_col2, tech_col3 = st.columns(3)

with tech_col1:
    st.markdown("""
    **Machine Learning:**
    <div class="tech-badge">TensorFlow</div>
    <div class="tech-badge">PyTorch</div>
    <div class="tech-badge">Scikit-learn</div>
    <div class="tech-badge">XGBoost</div>
    """, unsafe_allow_html=True)

with tech_col2:
    st.markdown("""
    **Data Processing:**
    <div class="tech-badge">Pandas</div>
    <div class="tech-badge">NumPy</div>
    <div class="tech-badge">Matplotlib</div>
    <div class="tech-badge">Plotly</div>
    """, unsafe_allow_html=True)

with tech_col3:
    st.markdown("""
    **Web Framework:**
    <div class="tech-badge">Streamlit</div>
    <div class="tech-badge">FastAPI</div>
    <div class="tech-badge">Docker</div>
    <div class="tech-badge">AWS</div>
    """, unsafe_allow_html=True)

# Company Information
st.markdown("""
<div class="company-info">
    <h2>🏢 MaterialTech Solutions</h2>
    <p><strong>Leading Innovation in Material Science Technology</strong></p>
    <p>Founded in 2020, MaterialTech Solutions is dedicated to advancing material science through 
    artificial intelligence and machine learning. Our mission is to accelerate scientific discovery 
    and enable breakthrough innovations in materials research.</p>
    <p>📍 San Francisco, CA | 🌐 www.materialtech-solutions.com</p>
</div>
""", unsafe_allow_html=True)

# Team Section
st.markdown('<h2 class="section-header">👥 Meet Our Team</h2>', unsafe_allow_html=True)

# Team members data
team_members = [
    {
        "name": "Dr. Sarah Chen",
        "role": "Project Lead & Data Scientist",
        "bio": "PhD in Materials Science from MIT. 8+ years experience in AI/ML applications for materials research.",
        "github": "https://github.com/sarahchen",
        "linkedin": "https://linkedin.com/in/sarahchen-materials",
        "email": "sarah.chen@materialtech.com"
    },
    {
        "name": "Alex Rodriguez",
        "role": "Machine Learning Engineer",
        "bio": "MS in Computer Science from Stanford. Specializes in deep learning and neural network architectures.",
        "github": "https://github.com/alexrodriguez",
        "linkedin": "https://linkedin.com/in/alex-rodriguez-ml",
        "email": "alex.rodriguez@materialtech.com"
    },
    {
        "name": "Dr. Michael Thompson",
        "role": "Materials Science Consultant",
        "bio": "Professor of Materials Engineering at UC Berkeley. Expert in computational materials science.",
        "github": "https://github.com/mthompson",
        "linkedin": "https://linkedin.com/in/michael-thompson-materials",
        "email": "michael.thompson@materialtech.com"
    },
    {
        "name": "Emma Johnson",
        "role": "Full Stack Developer",
        "bio": "BS in Software Engineering. Specializes in web development and user experience design.",
        "github": "https://github.com/emmajohnson",
        "linkedin": "https://linkedin.com/in/emma-johnson-dev",
        "email": "emma.johnson@materialtech.com"
    },
    {
        "name": "David Park",
        "role": "DevOps Engineer",
        "bio": "Expert in cloud infrastructure and deployment. Ensures scalable and reliable system architecture.",
        "github": "https://github.com/davidpark",
        "linkedin": "https://linkedin.com/in/david-park-devops",
        "email": "david.park@materialtech.com"
    }
]

# Display team members in cards
for i, member in enumerate(team_members):
    with st.container():
        col1, col2 = st.columns([1, 3])
        
        with col1:
            # Placeholder for profile image
            st.markdown(f"""
            <div style="width: 120px; height: 120px; border-radius: 50%; 
                       background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                       display: flex; align-items: center; justify-content: center; 
                       color: white; font-size: 2rem; font-weight: bold; margin: 1rem 0;">
                {member['name'].split()[0][0]}{member['name'].split()[1][0]}
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
                    📧 {member['email']}<br>
                    💼 <a href="{member['linkedin']}" target="_blank">LinkedIn Profile</a><br>
                    🐙 <a href="{member['github']}" target="_blank">GitHub Profile</a>
                </div>
            </div>
            """, unsafe_allow_html=True)

# Project Links
st.markdown('<h2 class="section-header">🔗 Project Links</h2>', unsafe_allow_html=True)

link_col1, link_col2, link_col3 = st.columns(3)

with link_col1:
    st.markdown("""
    ### 📱 Application
    - [Live Demo](https://smart-material-predictor.streamlit.app)
    - [User Documentation](https://docs.materialtech.com)
    - [API Reference](https://api.materialtech.com/docs)
    """)

with link_col2:
    st.markdown("""
    ### 🧪 Research
    - [Research Papers](https://research.materialtech.com)
    - [Dataset](https://data.materialtech.com)
    - [Benchmark Results](https://benchmark.materialtech.com)
    """)

with link_col3:
    st.markdown("""
    ### 💻 Development
    - [GitHub Repository](https://github.com/materialtech/smart-material-predictor)
    - [Issue Tracker](https://github.com/materialtech/smart-material-predictor/issues)
    - [Contributing Guide](https://github.com/materialtech/smart-material-predictor/blob/main/CONTRIBUTING.md)
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 2rem;">
    <p><strong>Smart Material Predictor</strong> © 2025 MaterialTech Solutions. All rights reserved.</p>
    <p>For inquiries, contact us at: info@materialtech-solutions.com</p>
</div>
""", unsafe_allow_html=True)

# Sidebar with quick navigation
st.sidebar.markdown("## 🧭 Quick Navigation")
st.sidebar.markdown("""
- [Project Overview](#project-overview)
- [Technology Stack](#technology-stack)
- [Our Company](#materialtech-solutions)
- [Meet Our Team](#meet-our-team)
- [Project Links](#project-links)
""")

st.sidebar.markdown("---")
st.sidebar.markdown("## 📞 Contact Us")
st.sidebar.markdown("""
**MaterialTech Solutions**  
📧 info@materialtech-solutions.com  
📱 +1 (555) 123-4567  
📍 San Francisco, CA  
""")

st.sidebar.markdown("---")
st.sidebar.markdown("## 🔗 Quick Links")
st.sidebar.markdown("""
- [GitHub](https://github.com/materialtech/smart-material-predictor)
- [Documentation](https://docs.materialtech.com)
- [LinkedIn](https://linkedin.com/company/materialtech-solutions)
- [Twitter](https://twitter.com/materialtech)
""")