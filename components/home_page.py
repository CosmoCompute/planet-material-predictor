import streamlit as st
import base64

def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    except Exception:
        return ""

def home():
    bg_b64 = get_base64_image("assets/images/hero_bg.png")
    
    # 1. Inject custom parent overrides so the iframe/content is edge-to-edge
    st.markdown("""
        <style>
            /* Make parent container borderless and edge-to-edge */
            .block-container {
                padding-top: 0rem !important;
                padding-bottom: 0rem !important;
                padding-left: 0rem !important;
                padding-right: 0rem !important;
                max-width: 100% !important;
            }
            /* Remove margins from vertical blocks */
            div[data-testid="stVerticalBlock"] > div {
                padding: 0 !important;
            }
            /* Ensure the iframe itself takes full width and fills height */
            iframe {
                width: 100% !important;
                height: 100vh !important;
                border: none !important;
                margin: 0 !important;
                padding: 0 !important;
            }
            /* Hide the main streamlit header/footer for home page to make it clean */
            header[data-testid="stHeader"] {
                background: transparent !important;
            }
            /* Hide the native button used for the callback */
            div[data-testid="stButton"] {
                display: none !important;
            }
        </style>
    """, unsafe_allow_html=True)

    # 2. Hidden native Streamlit button to catch the JavaScript click event
    if st.button("Start Analyzing", key="start_analyzing_btn"):
        st.session_state.redirect_to_temp_analysis = True
        st.rerun()

    html_content = f"""
    <!DOCTYPE html>
    <html class="dark h-full" lang="en">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <!-- Fonts -->
        <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@600;700;800&amp;family=Inter:wght@400;600&amp;display=swap" rel="stylesheet">
        <!-- Tailwind CSS -->
        <script src="https://cdn.tailwindcss.com?plugins=forms,container-queries"></script>
        <script>
            tailwind.config = {{
                darkMode: "class",
                theme: {{
                    extend: {{
                        "colors": {{
                            "surface": "#111415",
                            "on-surface": "#e1e3e4",
                        }},
                        "fontFamily": {{
                            "display-hero": ["Montserrat"],
                            "body-lg": ["Inter"]
                        }}
                    }}
                }}
            }};
        </script>
        
        <style>
            body, html {{
                margin: 0;
                padding: 0;
                height: 100%;
                overflow: hidden;
                font-family: 'Inter', sans-serif;
            }}
            .hero-background {{
                background-image: linear-gradient(rgba(17, 20, 21, 0.35), rgba(17, 20, 21, 0.75)), url('data:image/jpeg;base64,{bg_b64}');
                background-size: cover;
                background-position: center;
                height: 100%;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                position: relative;
            }}
            .text-glow {{
                text-shadow: 0 0 30px rgba(255, 255, 255, 0.2);
            }}
            /* Premium Button Styling */
            .premium-btn {{
                background: #ff6b00;
                color: #ffffff;
                font-family: 'Montserrat', sans-serif;
                font-weight: 700;
                font-size: 1.15rem;
                letter-spacing: 0.06em;
                text-transform: uppercase;
                padding: 1.25rem 3.5rem;
                border-radius: 8px;
                border: none;
                box-shadow: 0 10px 35px rgba(255, 107, 0, 0.4);
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                cursor: pointer;
                text-decoration: none;
                display: inline-block;
            }}
            .premium-btn:hover {{
                background: #ff8533;
                box-shadow: 0 15px 40px rgba(255, 107, 0, 0.6);
                transform: translateY(-2px);
            }}
            .premium-btn:active {{
                transform: translateY(-1px);
                box-shadow: 0 5px 15px rgba(255, 107, 0, 0.4);
            }}
            /* Top Brand Title */
            .brand-title {{
                position: absolute;
                top: 2.5rem;
                left: 3rem;
                font-family: 'Montserrat', sans-serif;
                font-weight: 800;
                font-size: 1.15rem;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                color: rgba(255, 255, 255, 0.95);
                z-index: 50;
            }}
        </style>
        
        <script>
            // Triggers the hidden parent Streamlit button to switch pages natively
            function triggerParentButton() {{
                try {{
                    const parentButtons = window.parent.document.querySelectorAll('button');
                    for (const btn of parentButtons) {{
                        if (btn.textContent.trim() === 'Start Analyzing') {{
                            btn.click();
                            break;
                        }}
                    }}
                }} catch (e) {{
                    console.error("Failed to trigger parent button:", e);
                }}
            }}
        </script>
    </head>
    <body class="h-full">
        <main class="h-full">
            <!-- Hero Section -->
            <section class="hero-background px-6 md:px-16 w-full text-center">
                <!-- Top-Left Brand Title -->
                <div class="brand-title">
                    Planetary Material Predictor
                </div>

                <div class="max-w-4xl mx-auto space-y-8 -mt-20">
                    <!-- Large Editorial Headline -->
                    <h1 class="font-display-hero text-5xl md:text-7xl text-white text-glow leading-tight tracking-tight font-extrabold">
                        Analyze the Universe,<br>One Layer at a Time
                    </h1>
                    <!-- Refined Subheader -->
                    <p class="font-body-lg text-lg md:text-xl text-[#cbd5e1] max-w-3xl mx-auto leading-relaxed">
                        Advanced Analysis &amp; Prediction Engine for Planetary Surface Composition. Precision data for the next generation of space exploration.
                    </p>
                    <!-- Prominent CTA -->
                    <div class="pt-6 flex justify-center">
                        <button onclick="triggerParentButton()" class="premium-btn">
                            Start Analyzing
                        </button>
                    </div>
                </div>
            </section>
        </main>
    </body>
    </html>
    """
    st.iframe(html_content, height=900)
