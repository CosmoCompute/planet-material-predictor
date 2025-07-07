import streamlit as st
from components import local_def

local_def.load_css("assets/style.css")

def notfound():
    st.markdown('<div class="not-found-container">', unsafe_allow_html=True)
    st.markdown('<h1 class="error-code">404</h1>', unsafe_allow_html=True)
    st.markdown('<h2 class="error-title">Page Not Found</h2>', unsafe_allow_html=True)
    st.markdown('''
    <p class="error-message">
        Oops! The page you're looking for seems to have wandered off. 
        Don't worry, it happens to the best of us. Let's get you back on track.
    </p>
    ''', unsafe_allow_html=True)
    st.markdown('<div class="search-container">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        search_query = st.text_input("", placeholder="Search for what you're looking for...", label_visibility="collapsed")
        if st.button("🔍 Search", key="search_btn"):
            if search_query:
                st.success(f"Searching for: {search_query}")
    st.markdown('</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🏠 Go Home", key="home_btn"):
            st.rerun()
    with col2:
        if st.button("📧 Contact Support", key="contact_btn"):
            st.info("Contact us at support@yourapp.com")
    st.markdown('''
    <div class="helpful-links">
        <h4>📚 Popular Pages</h4>
        <div class="link-grid">
            <div class="link-card">
                <div class="icon">🏠</div>
                <strong>Home</strong><br>
                <small>Return to main page</small>
            </div>
            <div class="link-card">
                <div class="icon">📊</div>
                <strong>Dashboard</strong><br>
                <small>View your analytics</small>
            </div>
            <div class="link-card">
                <div class="icon">⚙️</div>
                <strong>Settings</strong><br>
                <small>Manage preferences</small>
            </div>
            <div class="link-card">
                <div class="icon">❓</div>
                <strong>Help</strong><br>
                <small>Get assistance</small>
            </div>
        </div>
    </div>
    ''', unsafe_allow_html=True)
    with st.expander("ℹ️ What happened?"):
        st.write("""
        **Common reasons for this error:**
        - The page URL was typed incorrectly
        - The page has been moved or deleted
        - You followed an outdated link
        - The page is temporarily unavailable
        
        **What you can do:**
        - Check the URL for typos
        - Use the search box above
        - Navigate using the menu
        - Contact support if the problem persists
        """)
    st.markdown("---")
    st.markdown(
        '<p style="text-align: center; color: #7f8c8d; margin-top: 2rem;">Need help? We\'re here to assist you!</p>',
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)
