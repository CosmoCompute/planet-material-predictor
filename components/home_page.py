import streamlit as st
from PIL import Image

def home():
    # Set wide layout
    st.set_page_config(layout="wide", page_title="Planetary Material Insight")

    # Optional: Load and display a banner image
    banner = "https://images.unsplash.com/photo-1581090700227-1e7eafee7f81?fit=crop&w=1920&q=80"
    st.image(banner, use_container_width=True)

    # Custom title styling
    st.markdown("""
        <h1 style='text-align: center; color: #2c3e50; font-size: 3.5rem; margin-top: 20px;'>🌌 Planetary Material Insight</h1>
        <h3 style='text-align: center; color: #7f8c8d; font-weight: 400;'>By <strong>CosmoCompute</strong> | Smart Planetary Material Detection System</h3>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border: 1px solid #ccc;'>", unsafe_allow_html=True)

    # Intro section
    st.markdown("""
    <div style="text-align: justify; font-size: 1.2rem;">
        <p><strong>Planetary Material Insight</strong> is an intelligent system designed to predict and classify planetary surface materials using seismic and geophysical parameters. Our model leverages machine learning to assist planetary scientists in identifying rock types such as igneous, sedimentary, and metamorphic structures on Mars and other celestial bodies.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🌍 Key Features")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.success("**🔬 Material Classification**\n\nAI-powered rock and mineral classification from seismic data.")
    with col2:
        st.info("**📊 Insightful Analytics**\n\nVisual exploration of seismic velocities, wave patterns, and more.")
    with col3:
        st.warning("**🚀 Space Application Ready**\n\nOptimized for Mars, Moon, and other extraterrestrial missions.")

    st.markdown("<br><br>", unsafe_allow_html=True)

    st.markdown("""
        <div style='text-align: center; font-size: 1.1rem;'>
            Want to explore how AI meets planetary science?<br>
            Navigate using the sidebar and dive into the cosmos 🌠
        </div>
    """, unsafe_allow_html=True)
