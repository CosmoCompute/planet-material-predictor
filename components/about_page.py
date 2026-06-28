import streamlit as st
import os
import base64

def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    except Exception:
        return ""

def about_us():
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
                height: 1100px !important;
                border: none !important;
                margin: 0 !important;
                padding: 0 !important;
            }
            /* Hide the main streamlit header/footer for home page to make it clean */
            header[data-testid="stHeader"] {
                background: transparent !important;
            }</style>
    """, unsafe_allow_html=True)

    # Base64-encode local team image
    swarnabha_b64 = get_base64_image("assets/team/10001314851.jpg")
    arijit_img_url = "https://avatars.githubusercontent.com/u/143516210?v=4"
    swarnabha_img_url = f"data:image/jpeg;base64,{swarnabha_b64}" if swarnabha_b64 else "https://via.placeholder.com/400"

    html_content = f"""
    <!DOCTYPE html>
    <html class="dark" lang="en">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <!-- Fonts -->
        <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;700;800&amp;family=Inter:wght@400;600&amp;display=swap" rel="stylesheet">
        <link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&amp;display=swap" rel="stylesheet">
        <!-- Tailwind CSS -->
        <script src="https://cdn.tailwindcss.com?plugins=forms,container-queries"></script>
        <script>
            tailwind.config = {{
                darkMode: "class",
                theme: {{
                    extend: {{
                        "colors": {{
                            "surface-tint": "#ffb693",
                            "primary-fixed": "#ffdbcc",
                            "on-secondary": "#1e2084",
                            "on-primary-container": "#572000",
                            "surface": "#111415",
                            "on-error-container": "#ffdad6",
                            "error": "#ffb4ab",
                            "outline": "#a98a7d",
                            "inverse-primary": "#a04100",
                            "secondary-container": "#373a9b",
                            "on-primary-fixed-variant": "#7a3000",
                            "tertiary-container": "#9896b0",
                            "on-tertiary": "#2f2e43",
                            "surface-container-lowest": "#0c0f10",
                            "secondary-fixed": "#e1e0ff",
                            "on-surface-variant": "#e2bfb0",
                            "surface-container-high": "#282a2b",
                            "background": "#111415",
                            "on-primary": "#561f00",
                            "inverse-surface": "#e1e3e4",
                            "on-tertiary-container": "#2f2f44",
                            "tertiary-fixed-dim": "#c6c4df",
                            "tertiary": "#c6c4df",
                            "surface-dim": "#111415",
                            "on-surface": "#e1e3e4",
                            "on-error": "#690005",
                            "secondary-fixed-dim": "#c0c1ff",
                            "primary": "#ffb693",
                            "secondary": "#c0c1ff",
                            "on-secondary-fixed": "#04006d",
                            "on-tertiary-fixed-variant": "#45455b",
                            "on-primary-fixed": "#351000",
                            "on-secondary-container": "#abaeff",
                            "surface-variant": "#323536",
                            "surface-container-highest": "#323536",
                            "on-secondary-fixed-variant": "#373a9b",
                            "on-background": "#e1e3e4",
                            "primary-container": "#ff6b00",
                            "inverse-on-surface": "#2e3132",
                            "tertiary-fixed": "#e2e0fc",
                            "outline-variant": "#5a4136",
                            "on-tertiary-fixed": "#1a1a2d",
                            "primary-fixed-dim": "#ffb693",
                            "surface-bright": "#373a3b",
                            "error-container": "#93000a",
                            "surface-container-low": "#191c1d",
                            "surface-container": "#1d2021"
                        }},
                        "borderRadius": {{
                            "DEFAULT": "0.125rem",
                            "lg": "0.25rem",
                            "xl": "0.5rem",
                            "full": "0.75rem"
                        }},
                        "spacing": {{
                            "gutter": "32px",
                            "margin-desktop": "64px",
                            "unit-lg": "32px",
                            "unit-md": "16px",
                            "unit-xl": "64px",
                            "unit-hero": "128px",
                            "margin-mobile": "24px",
                            "container-max": "1280px",
                            "unit-xs": "4px",
                            "unit-sm": "8px"
                        }},
                        "fontFamily": {{
                            "headline-md": ["Montserrat"],
                            "display-hero": ["Montserrat"],
                            "headline-lg": ["Montserrat"],
                            "headline-lg-mobile": ["Montserrat"],
                            "body-md": ["Inter"],
                            "label-sm": ["Inter"],
                            "body-lg": ["Inter"]
                        }},
                        "fontSize": {{
                            "headline-md": ["32px", {{"lineHeight": "1.3", "fontWeight": "600"}}],
                            "display-hero": ["72px", {{"lineHeight": "1.1", "letterSpacing": "-0.04em", "fontWeight": "800"}}],
                            "headline-lg": ["48px", {{"lineHeight": "1.2", "letterSpacing": "-0.02em", "fontWeight": "700"}}],
                            "headline-lg-mobile": ["36px", {{"lineHeight": "1.2", "fontWeight": "700"}}],
                            "body-md": ["16px", {{"lineHeight": "1.5", "fontWeight": "400"}}],
                            "label-sm": ["14px", {{"lineHeight": "1", "letterSpacing": "0.05em", "fontWeight": "600"}}],
                            "body-lg": ["20px", {{"lineHeight": "1.6", "letterSpacing": "-0.01em", "fontWeight": "400"}}]
                        }}
                    }}
                }},
            }};
        </script>
        <style>
            body {{
                background-color: #111415;
                color: #e1e3e4;
                font-family: 'Inter', sans-serif;
            }}

            .glass-card {{
                background: rgba(255, 255, 255, 0.02);
                backdrop-filter: blur(12px);
                border: 1px solid rgba(255, 255, 255, 0.06);
                transition: all 0.6s cubic-bezier(0.4, 0, 0.2, 1);
                position: relative;
            }}

            .glass-card:hover {{
                transform: translateY(-8px) scale(1.03);
                border-color: rgba(255, 107, 0, 0.4);
                box-shadow: 0 20px 40px -10px rgba(0, 0, 0, 0.5), 0 0 30px rgba(255, 107, 0, 0.15);
                z-index: 10;
            }}

            .glass-card .team-image {{
                transition: transform 0.8s cubic-bezier(0.4, 0, 0.2, 1), filter 0.8s cubic-bezier(0.4, 0, 0.2, 1);
            }}

            .glass-card:hover .team-image {{
                transform: scale(1.08);
                filter: grayscale(0);
            }}

            .team-image-container::after {{
                content: '';
                position: absolute;
                inset: 0;
                background: linear-gradient(to bottom, transparent 65%, #111415 100%);
                opacity: 0.9;
                pointer-events: none;
            }}

            .social-icon {{
                transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
                display: inline-flex;
                align-items: center;
                justify-content: center;
                width: 40px;
                height: 40px;
                border-radius: 50%;
                background: rgba(255, 255, 255, 0.03);
                border: 1px solid rgba(255, 255, 255, 0.05);
                color: #e2bfb0;
            }}

            .social-icon:hover {{
                transform: translateY(-4px);
                color: #ffb693;
                background: rgba(255, 107, 0, 0.1);
                border-color: rgba(255, 107, 0, 0.3);
            }}

            .reveal-content {{
                opacity: 0.85;
                transition: all 0.6s cubic-bezier(0.4, 0, 0.2, 1);
            }}

            @keyframes fadeInUp {{
                from {{
                    opacity: 0;
                    transform: translateY(30px);
                }}
                to {{
                    opacity: 1;
                    transform: translateY(0);
                }}
            }}

            .animate-fade-in-up {{
                animation: fadeInUp 1s cubic-bezier(0.4, 0, 0.2, 1) forwards;
                opacity: 0;
            }}

            .delay-1 {{ animation-delay: 0.2s; }}
            .delay-2 {{ animation-delay: 0.4s; }}

            @media (prefers-reduced-motion: reduce) {{
                .glass-card, .team-image, .social-icon, .reveal-content, .animate-fade-in-up {{
                    transition: none !important;
                    animation: none !important;
                    transform: none !important;
                    opacity: 1 !important;
                }}
            }}

            .material-symbols-outlined {{
                font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24;
            }}
        </style>
    </head>
    <body class="bg-background text-on-background overflow-x-hidden">
        <main class="relative pt-12 pb-unit-xl">
            <!-- Background Atmospheric Elements -->
            <div class="absolute top-0 right-0 w-1/2 h-[600px] bg-secondary/5 blur-[120px] rounded-full -translate-y-1/2 translate-x-1/4 pointer-events-none"></div>
            <div class="absolute bottom-0 left-0 w-1/3 h-[500px] bg-primary/5 blur-[120px] rounded-full translate-y-1/4 -translate-x-1/4 pointer-events-none"></div>
            
            <!-- Hero Content -->
            <section class="max-w-container-max mx-auto px-margin-mobile md:px-margin-desktop text-center mb-16">
                <div class="inline-flex items-center gap-unit-xs px-unit-md py-unit-xs rounded-full border border-primary/20 bg-primary/5 text-primary mb-6">
                    <span class="material-symbols-outlined text-[18px]">group</span>
                    <span class="font-label-sm text-label-sm uppercase tracking-widest">Our Cosmosompute Team</span>
                </div>
                <h1 class="font-headline-lg-mobile md:font-headline-lg text-headline-lg-mobile md:text-headline-lg mb-6 max-w-3xl mx-auto">
                    The Minds Behind the Science
                </h1>
                <p class="font-body-lg text-body-lg text-[#cbd5e1] max-w-2xl mx-auto opacity-90 leading-relaxed">
                    A dedicated team of computational scientists and data engineers pushing the boundaries of planetary material prediction through high-fidelity analytics and ML innovation.
                </p>
            </section>
            
            <!-- Team Grid -->
            <section class="max-w-container-max mx-auto px-margin-mobile md:px-margin-desktop grid grid-cols-1 md:grid-cols-2 gap-unit-xl">
                <!-- Leader Card: Arijit Chowdhury -->
                <div class="glass-card rounded-xl overflow-hidden group animate-fade-in-up delay-1">
                    <div class="relative team-image-container aspect-[4/5] overflow-hidden">
                        <img alt="Arijit Chowdhury" class="team-image w-full h-full object-cover grayscale" src="{arijit_img_url}"/>
                        <div class="absolute inset-0 bg-gradient-to-t from-background via-transparent to-transparent opacity-60"></div>
                    </div>
                    <div class="p-unit-lg relative">
                        <div class="absolute -top-6 right-6">
                            <span class="bg-[#ff6b00] text-white font-label-sm text-label-sm px-4 py-1.5 rounded-full shadow-lg border border-white/10">LEADER</span>
                        </div>
                        <h3 class="font-headline-md text-headline-md text-on-surface mb-2">Arijit Chowdhury</h3>
                        <div class="flex items-center gap-unit-sm mb-4">
                            <div class="h-[2px] w-12 bg-[#ff6b00]"></div>
                            <span class="font-label-sm text-label-sm text-primary uppercase tracking-widest">Project Visionary &amp; Lead Developer</span>
                        </div>
                        <div class="reveal-content space-y-4">
                            <div class="space-y-2 font-body-md text-body-md text-on-surface-variant leading-relaxed">
                                <p><strong class="text-on-surface">Bio:</strong> A B.Sc Student at University of Calcutta, specializing in Computer Science.</p>
                                <p><strong class="text-on-surface">Contribution:</strong> Frontend UI, Data Analysis, Visualization, Backend Logic, Model Training, &amp; Database Management.</p>
                            </div>
                            <div class="flex items-center gap-unit-md pt-2">
                                <a class="social-icon" href="https://github.com/student-Arijit" target="_blank" title="GitHub">
                                    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="currentColor" viewBox="0 0 16 16">
                                        <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27s1.36.09 2 .27c1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.28.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21-.15.46-.55.38A8.01 8.01 0 0 1 0 8z"/>
                                    </svg>
                                </a>
                                <a class="social-icon" href="mailto:arijitchowdhury4467@gmail.com" title="Email">
                                    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                                        <path d="M22 6c0-1.1-.9-2-2-2H4c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V6zm-2 0-8 5-8-5h16zm0 12H4V8l8 5 8-5v10z"></path>
                                    </svg>
                                </a>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Co-Leader Card: Swarnabha Halder -->
                <div class="glass-card rounded-xl overflow-hidden group animate-fade-in-up delay-2">
                    <div class="relative team-image-container aspect-[4/5] overflow-hidden">
                        <img alt="Swarnabha Halder" class="team-image w-full h-full object-cover grayscale" src="{swarnabha_img_url}"/>
                        <div class="absolute inset-0 bg-gradient-to-t from-background via-transparent to-transparent opacity-60"></div>
                    </div>
                    <div class="p-unit-lg relative">
                        <div class="absolute -top-6 right-6">
                            <span class="bg-[#323536] text-white font-label-sm text-label-sm px-4 py-1.5 rounded-full shadow-lg border border-white/10 group-hover:bg-[#ff6b00] group-hover:text-white transition-colors duration-500">CO-LEADER</span>
                        </div>
                        <h3 class="font-headline-md text-headline-md text-on-surface mb-2">Swarnabha Halder</h3>
                        <div class="flex items-center gap-unit-sm mb-4">
                            <div class="h-[2px] w-12 bg-on-surface-variant group-hover:bg-[#ff6b00] transition-colors duration-500"></div>
                            <span class="font-label-sm text-label-sm text-on-surface-variant group-hover:text-primary transition-colors duration-500 uppercase tracking-widest">Technical Architect</span>
                        </div>
                        <div class="reveal-content space-y-4">
                            <div class="space-y-2 font-body-md text-body-md text-on-surface-variant leading-relaxed">
                                <p><strong class="text-on-surface">Bio:</strong> A B.Tech in Computer Science with a specialization in Data Science from SMIT.</p>
                                <p><strong class="text-on-surface">Contribution:</strong> UI Design, Data Analysis, Model Training, &amp; Core Logic Building.</p>
                            </div>
                            <div class="flex items-center gap-unit-md pt-2">
                                <a class="social-icon" href="https://github.com/swarnabha-dev" target="_blank" title="GitHub">
                                    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="currentColor" viewBox="0 0 16 16">
                                        <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27s1.36.09 2 .27c1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.28.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21-.15.46-.55.38A8.01 8.01 0 0 1 0 8z"/>
                                    </svg>
                                </a>
                                <a class="social-icon" href="mailto:swarnabhahalder80137@gmail.com" title="Email">
                                    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                                        <path d="M22 6c0-1.1-.9-2-2-2H4c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V6zm-2 0-8 5-8-5h16zm0 12H4V8l8 5 8-5v10z"></path>
                                    </svg>
                                </a>
                            </div>
                        </div>
                    </div>
                </div>
            </section>
        </main>
        
        <!-- Footer -->
        <footer class="bg-surface-container-lowest border-t border-white/5 mt-16">
            <div class="flex flex-col md:flex-row justify-between items-center px-margin-mobile md:px-margin-desktop py-unit-lg w-full max-w-container-max mx-auto gap-unit-md">
                <div class="flex flex-col items-center md:items-start gap-unit-xs">
                    <span class="font-headline-md text-headline-md text-on-surface">Planet Material Predictor</span>
                    <p class="font-label-sm text-label-sm text-on-surface-variant opacity-60">© 2024 Planet Material Predictor. High-fidelity planetary analytics.</p>
                </div>
                <div class="flex flex-wrap justify-center gap-unit-md">
                    <a class="font-label-sm text-label-sm text-on-surface-variant hover:text-primary transition-colors" href="#">Privacy Policy</a>
                    <a class="font-label-sm text-label-sm text-on-surface-variant hover:text-primary transition-colors" href="#">Terms of Service</a>
                    <a class="font-label-sm text-label-sm text-on-surface-variant hover:text-primary transition-colors" href="#">Contact Support</a>
                    <a class="font-label-sm text-label-sm text-on-surface-variant hover:text-primary transition-colors" href="#">Documentation</a>
                </div>
            </div>
        </footer>
        
        <script>
            // High-fidelity micro-interactions (3D Card Tilt)
            document.querySelectorAll('.glass-card').forEach(card => {{
                card.addEventListener('mousemove', (e) => {{
                    const rect = card.getBoundingClientRect();
                    const x = e.clientX - rect.left;
                    const y = e.clientY - rect.top;
                    
                    const centerX = rect.width / 2;
                    const centerY = rect.height / 2;
                    const rotateX = (y - centerY) / 25;
                    const rotateY = (centerX - x) / 25;

                    if (!window.matchMedia('(prefers-reduced-motion: reduce)').matches) {{
                        card.style.transform = `translateY(-8px) scale(1.03) perspective(1000px) rotateX(${{rotateX}}deg) rotateY(${{rotateY}}deg)`;
                    }}
                }});

                card.addEventListener('mouseleave', () => {{
                    if (!window.matchMedia('(prefers-reduced-motion: reduce)').matches) {{
                        card.style.transform = '';
                    }}
                }});
            }});
        </script>
    </body>
    </html>
    """
    st.iframe(html_content, height=1050)
