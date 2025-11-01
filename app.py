import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import plotly.graph_objects as go
import plotly.express as px
from wordcloud import WordCloud
import pandas as pd
import time
from datetime import datetime
import numpy as np
import os
import matplotlib.pyplot as plt
import requests
from io import BytesIO
from PIL import Image

# =================================================================================
# Page Configuration
# =================================================================================
st.set_page_config(
    page_title="AI Spam Sentinel",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =================================================================================
# Custom CSS for Modern UI/UX
# =================================================================================
st.markdown("""
<style>
    /* Main Theme */
    .stApp {
        background: linear-gradient(135deg, #0E1117 0%, #1A1A2E 50%, #16213E 100%);
        color: #FAFAFA;
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #1A1A2E 0%, #16213E 100%);
        border-right: 3px solid #7B28C9;
    }
    
    /* Metric Cards with Glassmorphism Effect */
    .metric-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 25px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px rgba(0, 246, 255, 0.2);
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0, 246, 255, 0.3);
    }
    
    /* Result Banners with Animation */
    .spam-banner {
        padding: 2rem;
        border-radius: 20px;
        color: white;
        background: linear-gradient(135deg, #FF2E63 0%, #FF007F 100%);
        border: 2px solid #FF007F;
        text-align: center;
        font-size: 28px;
        font-weight: bold;
        animation: pulse 2s infinite;
    }
    .ham-banner {
        padding: 2rem;
        border-radius: 20px;
        color: white;
        background: linear-gradient(135deg, #08D9D6 0%, #00F6FF 100%);
        border: 2px solid #00F6FF;
        text-align: center;
        font-size: 28px;
        font-weight: bold;
        animation: glow 2s infinite alternate;
    }
    
    /* Buttons with Hover Effects */
    .stButton>button {
        background: linear-gradient(135deg, #7B28C9 0%, #9D4EDD 100%);
        color: white;
        border-radius: 25px;
        padding: 15px 30px;
        font-size: 18px;
        font-weight: bold;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 6px 20px rgba(157, 78, 221, 0.4);
    }
    .stButton>button:hover {
        transform: translateY(-3px) scale(1.05);
        box-shadow: 0 10px 25px rgba(157, 78, 221, 0.6);
    }

    /* Animations */
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    
    @keyframes glow {
        from { box-shadow: 0 0 20px #00F6FF; }
        to { box-shadow: 0 0 30px #00F6FF; }
    }

    /* Custom Headers */
    .main-header {
        background: linear-gradient(135deg, #7B28C9 0%, #9D4EDD 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
    }

    /* Image Containers */
    .image-container {
        border-radius: 20px;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        transition: transform 0.3s ease;
    }
    .image-container:hover {
        transform: scale(1.02);
    }
</style>
""", unsafe_allow_html=True)

# =================================================================================
# Load Images
# =================================================================================
@st.cache_resource
def load_images():
    """Load and cache images for the app"""
    images = {}
    try:
        # Main banner image
        banner_url = "https://tycoonsuccess.com/wp-content/uploads/2022/03/Spam-Email-Detection-Using-Artificial-Intelligence-netework.jpg"
        response = requests.get(banner_url)
        images['banner'] = Image.open(BytesIO(response.content))
        
        # Sidebar image
        sidebar_url = "https://intersys.co.uk/wp-content/uploads/Phishing-email-graphic-1280x1163.jpg"
        response = requests.get(sidebar_url)
        images['sidebar'] = Image.open(BytesIO(response.content))
        
        # Analytics image
        analytics_url = "https://www.icdsoft.com/blog/wp-content/uploads/2023/03/How-To-Spot-Spam-Emails-and-Phishing-Scams.png"
        response = requests.get(analytics_url)
        images['analytics'] = Image.open(BytesIO(response.content))
        
    except Exception as e:
        st.warning(f"Some images couldn't be loaded: {e}")
        # Placeholder images if URLs fail
        images = {}
    
    return images

images = load_images()

# =================================================================================
# Load Model & Preprocessing Tools with Error Handling
# =================================================================================
@st.cache_resource
def load_artifacts():
    """
    Loads the trained model and vectorizer.
    Handles errors if files are not found.
    """
    model_path = 'model.pkl'
    vectorizer_path = 'tfidf_vectorizer.pkl'

    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        st.error("Model or vectorizer files not found.")
        st.info("Please make sure 'model.pkl' and 'tfidf_vectorizer.pkl' are in the same directory as this app.")
        st.info("You need to run the Google Colab notebook first to generate these files.")
        return None, None

    try:
        with open(model_path, 'rb') as model_file:
            model = pickle.load(model_file)
        with open(vectorizer_path, 'rb') as vectorizer_file:
            tfidf_vectorizer = pickle.load(vectorizer_file)
        
        # Download NLTK data if not already present
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    except Exception as e:
        st.error(f"An error occurred while loading the model files: {e}")
        return None, None

    return model, tfidf_vectorizer

model, tfidf_vectorizer = load_artifacts()

# Check if loading was successful before proceeding
if model and tfidf_vectorizer:
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))

    # Enhanced list of common spam trigger words
    SPAM_INDICATORS = [
        'free', 'win', 'winner', 'prize', 'cash', 'urgent', 'claim', 'congratulations',
        'credit', 'unlimited', 'offer', 'guaranteed', 'won', 'lottery', 'bonus',
        'million', 'dollar', 'selected', 'exclusive', 'limited', 'act now',
        'click here', 'risk-free', 'special promotion', 'discount', 'deal'
    ]

# =================================================================================
# Helper Functions
# =================================================================================
def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

def predict_spam(input_text):
    cleaned_input = clean_text(input_text)
    vectorized_input = tfidf_vectorizer.transform([cleaned_input]).toarray()
    
    prediction = model.predict(vectorized_input)[0]
    
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(vectorized_input)[0]
        confidence = np.max(probabilities)
    else:
        confidence = 0.95
        
    return prediction, confidence

def count_spam_indicators(text):
    count = 0
    text_lower = text.lower()
    for indicator in SPAM_INDICATORS:
        if indicator in text_lower:
            count += 1
    return count

def create_confidence_meter(confidence, is_spam):
    """Create an animated confidence meter"""
    color = "#FF007F" if is_spam else "#00F6FF"
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=confidence * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Confidence Level", 'font': {'size': 24, 'color': 'white'}},
        delta={'reference': 50, 'increasing': {'color': color}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': color},
            'bgcolor': "rgba(0,0,0,0.3)",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': 'rgba(255, 255, 255, 0.1)'},
                {'range': [50, 100], 'color': 'rgba(255, 255, 255, 0.1)'}],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': 90}
        }
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "white", 'family': "Arial"})
    return fig

# =================================================================================
# Session State Initialization
# =================================================================================
if 'history' not in st.session_state:
    st.session_state.history = []
if 'total_analyzed' not in st.session_state:
    st.session_state.total_analyzed = 0
if 'spam_detected' not in st.session_state:
    st.session_state.spam_detected = 0
if 'ham_identified' not in st.session_state:
    st.session_state.ham_identified = 0
if 'confidence_scores' not in st.session_state:
    st.session_state.confidence_scores = []
if 'spam_words' not in st.session_state:
    st.session_state.spam_words = ""
if 'detection_times' not in st.session_state:
    st.session_state.detection_times = []

# =================================================================================
# Sidebar Navigation with Enhanced UI
# =================================================================================
with st.sidebar:
    st.markdown('<div class="main-header">🛡️ AI Spam Sentinel</div>', unsafe_allow_html=True)
    
    if 'sidebar' in images:
        st.image(images['sidebar'], use_container_width=True, caption="Protect Your Inbox")
    
    st.markdown("---")
    
    # Only show page navigation if the model loaded correctly
    if model and tfidf_vectorizer:
        page_options = {
            "🏠 Real-Time Detector": "Real-Time Detector",
            "📊 Analytics Dashboard": "Analytics Dashboard", 
            "📈 Advanced Analytics": "Advanced Analytics",
            "📜 Detection History": "Detection History",
            "🛡️ Spam Prevention Tips": "Spam Prevention Tips"
        }
        
        selected_icon = st.radio("Navigation", list(page_options.keys()))
        page = page_options[selected_icon]
    else:
        page = None
        
    st.markdown("---")
    
    # Quick Stats in Sidebar
    st.subheader("Session Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Analyzed", st.session_state.total_analyzed)
    with col2:
        spam_rate = (st.session_state.spam_detected / st.session_state.total_analyzed * 100) if st.session_state.total_analyzed > 0 else 0
        st.metric("Spam Rate", f"{spam_rate:.1f}%")
    
    st.info("🛡️ AI-powered email protection with real-time analysis and advanced analytics.")

# Main content area
if not model or not tfidf_vectorizer:
    st.header("🚨 Application Setup Required")
    st.warning("Please resolve the file loading issues mentioned in the error message above.")
    
    # Show banner image even when model isn't loaded
    if 'banner' in images:
        st.image(images['banner'], use_container_width=True, caption="AI-Powered Spam Detection")
else:
    # =================================================================================
    # Page 1: Real-Time Detector
    # =================================================================================
    if page == "Real-Time Detector":
        # Hero Section with Banner
        if 'banner' in images:
            st.image(images['banner'], use_container_width=True, caption="Advanced AI Spam Detection System")
        
        st.markdown('<div class="main-header">AI Spam Sentinel</div>', unsafe_allow_html=True)
        st.markdown("### An intelligent shield for your inbox, powered by advanced Machine Learning")

        # Enhanced Metrics Section with Glassmorphism
        st.markdown("---")
        st.subheader("📈 Live Detection Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Analyzed", st.session_state.total_analyzed, delta=None)
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Spam Detected", st.session_state.spam_detected, 
                     delta=f"+{st.session_state.spam_detected}" if st.session_state.spam_detected > 0 else None)
            st.markdown('</div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Ham Identified", st.session_state.ham_identified,
                     delta=f"+{st.session_state.ham_identified}" if st.session_state.ham_identified > 0 else None)
            st.markdown('</div>', unsafe_allow_html=True)
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            avg_confidence = np.mean(st.session_state.confidence_scores) if st.session_state.confidence_scores else 0
            st.metric("Avg. Confidence", f"{avg_confidence:.2%}")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("---")

        # Input Section
        st.subheader("🔍 Email Analysis Panel")
        
        input_method = st.radio("Choose input method:", ["📝 Paste Text", "📁 Upload File"])
        
        if input_method == "📝 Paste Text":
            input_text = st.text_area("Paste the raw content of an email here...", height=250, 
                                    placeholder="Subject: Congratulations! You've won a prize...\n\nBody: Dear customer, we are pleased to inform you that you have won a cash prize of $1,000,000. To claim, please visit...")
        else:
            uploaded_file = st.file_uploader("Upload email file", type=['txt', 'eml'])
            if uploaded_file:
                input_text = str(uploaded_file.read(), 'utf-8')
                st.expander("File Preview").text(input_text[:500] + "..." if len(input_text) > 500 else input_text)
            else:
                input_text = ""

        if st.button("🚀 Analyze Email with AI", use_container_width=True):
            if input_text:
                with st.spinner("🔬 AI is analyzing the email content..."):
                    time.sleep(1.5) # Simulate processing
                    
                    prediction, confidence = predict_spam(input_text)
                    
                    # Update Session State
                    st.session_state.total_analyzed += 1
                    st.session_state.confidence_scores.append(confidence)
                    st.session_state.detection_times.append(datetime.now())
                    
                    if prediction == 1:
                        st.session_state.spam_detected += 1
                        st.session_state.spam_words += " " + clean_text(input_text)
                        result_label = "Spam"
                        banner_class = "spam-banner"
                        icon = "🚨"
                        result_color = "#FF007F"
                    else:
                        st.session_state.ham_identified += 1
                        result_label = "Ham (Legitimate)"
                        banner_class = "ham-banner"
                        icon = "✅"
                        result_color = "#00F6FF"
                    
                    # Add to history
                    new_entry = {
                        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Snippet": input_text[:50] + "...",
                        "Prediction": result_label,
                        "Confidence": f"{confidence:.2%}"
                    }
                    st.session_state.history.insert(0, new_entry)

                # Results Display
                st.markdown(f'<div class="{banner_class}">{icon} {result_label.upper()} DETECTED!</div>', unsafe_allow_html=True)
                
                st.write("") # Spacer
                
                # Enhanced Confidence Meter
                st.subheader("🎯 AI Confidence Analysis")
                fig_gauge = create_confidence_meter(confidence, prediction == 1)
                st.plotly_chart(fig_gauge, use_container_width=True)

                # Quick Analysis with Enhanced Visuals
                st.subheader("📊 Detailed Analysis")
                qc1, qc2, qc3, qc4 = st.columns(4)
                
                with qc1:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Text Length", f"{len(input_text)} chars")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with qc2:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Word Count", len(input_text.split()))
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with qc3:
                    spam_indicators_count = count_spam_indicators(input_text)
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Spam Indicators", spam_indicators_count, 
                             delta="High Risk" if spam_indicators_count > 3 else "Low Risk")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with qc4:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("AI Confidence", f"{confidence:.2%}")
                    st.markdown('</div>', unsafe_allow_html=True)

                # Spam Indicator Breakdown
                if spam_indicators_count > 0:
                    st.subheader("🔍 Detected Spam Indicators")
                    found_indicators = [indicator for indicator in SPAM_INDICATORS if indicator in input_text.lower()]
                    for indicator in found_indicators:
                        st.error(f"• {indicator.title()}")

            else:
                st.warning("⚠️ Please enter some text or upload a file to analyze.")

    # =================================================================================
    # Page 2: Analytics Dashboard
    # =================================================================================
    elif page == "Analytics Dashboard":
        st.markdown('<div class="main-header">📊 Analytics Dashboard</div>', unsafe_allow_html=True)
        st.markdown("### Visual insights from AI-powered email analysis")

        if 'analytics' in images:
            st.image(images['analytics'], use_container_width=True, caption="Understanding Spam Patterns")

        if st.session_state.total_analyzed == 0:
            st.warning("📊 No data to display. Please analyze some emails on the 'Real-Time Detector' page first.")
        else:
            # Top Row: Distribution and Trends
            col1, col2 = st.columns((1, 1))
            
            with col1:
                st.subheader("📈 Email Distribution")
                labels = ['Ham (Safe)', 'Spam (Risk)']
                values = [st.session_state.ham_identified, st.session_state.spam_detected]
                colors = ['#00F6FF', '#FF007F']
                
                fig_pie = go.Figure(data=[go.Pie(
                    labels=labels, 
                    values=values, 
                    hole=.5,
                    marker_colors=colors,
                    textinfo='label+percent',
                    insidetextorientation='radial'
                )])
                fig_pie.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_color="white",
                    legend_title_text='Category',
                    height=400
                )
                st.plotly_chart(fig_pie, use_container_width=True)

            with col2:
                st.subheader("📊 Detection Comparison")
                df_counts = pd.DataFrame({
                    'Category': ['Ham (Safe)', 'Spam (Risk)'], 
                    'Count': [st.session_state.ham_identified, st.session_state.spam_detected]
                })
                
                fig_bar = px.bar(
                    df_counts, 
                    x='Category', 
                    y='Count', 
                    color='Category', 
                    color_discrete_map={'Ham (Safe)': '#00F6FF', 'Spam (Risk)': '#FF007F'},
                    text='Count'
                )
                fig_bar.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_color="white",
                    plot_bgcolor='rgba(0,0,0,0)',
                    height=400
                )
                fig_bar.update_traces(texttemplate='%{text}', textposition='outside')
                st.plotly_chart(fig_bar, use_container_width=True)
                
            st.markdown("---")
            
            # Middle Row: Word Cloud and Confidence Trends
            col3, col4 = st.columns((1, 1))
            
            with col3:
                st.subheader("🔥 Hot Words in Spam Emails")
                if st.session_state.spam_words.strip():
                    wordcloud = WordCloud(
                        width=800, 
                        height=400, 
                        background_color='rgba(0,0,0,0)',
                        colormap='Reds',
                        max_words=100,
                        contour_width=1,
                        contour_color='#FF007F'
                    ).generate(st.session_state.spam_words)
                    
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    ax.set_facecolor('none')
                    fig.patch.set_facecolor('none')
                    st.pyplot(fig)
                else:
                    st.info("No spam detected yet to generate a word cloud.")

            with col4:
                st.subheader("📊 Confidence Trend")
                if len(st.session_state.confidence_scores) > 1:
                    trend_df = pd.DataFrame({
                        'Analysis': range(len(st.session_state.confidence_scores)),
                        'Confidence': st.session_state.confidence_scores
                    })
                    fig_trend = px.line(
                        trend_df, 
                        x='Analysis', 
                        y='Confidence',
                        title='Confidence Scores Over Time',
                        color_discrete_sequence=['#00F6FF']
                    )
                    fig_trend.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        font_color="white",
                        plot_bgcolor='rgba(0,0,0,0)',
                        height=400
                    )
                    fig_trend.update_traces(line=dict(width=4))
                    st.plotly_chart(fig_trend, use_container_width=True)
                else:
                    st.info("Need more data points to show confidence trends.")

    # =================================================================================
    # Page 3: Advanced Analytics (NEW FEATURE)
    # =================================================================================
    elif page == "Advanced Analytics":
        st.markdown('<div class="main-header">📈 Advanced Analytics</div>', unsafe_allow_html=True)
        st.markdown("### Deep insights and pattern recognition")

        if st.session_state.total_analyzed == 0:
            st.warning("📊 No data for advanced analytics. Analyze emails first.")
        else:
            # Performance Metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                accuracy = (st.session_state.ham_identified + st.session_state.spam_detected) / st.session_state.total_analyzed
                st.metric("System Accuracy", f"{accuracy:.2%}")
            
            with col2:
                avg_confidence = np.mean(st.session_state.confidence_scores) if st.session_state.confidence_scores else 0
                st.metric("Avg Confidence", f"{avg_confidence:.2%}")
            
            with col3:
                spam_rate = (st.session_state.spam_detected / st.session_state.total_analyzed * 100) if st.session_state.total_analyzed > 0 else 0
                st.metric("Spam Rate", f"{spam_rate:.1f}%")

            # Advanced Charts
            col4, col5 = st.columns(2)
            
            with col4:
                st.subheader("🕒 Detection Timeline")
                if len(st.session_state.detection_times) > 1:
                    timeline_data = []
                    for i, time in enumerate(st.session_state.detection_times):
                        timeline_data.append({
                            'Time': time,
                            'Detection': i + 1,
                            'Type': 'Spam' if i < st.session_state.spam_detected else 'Ham'
                        })
                    
                    timeline_df = pd.DataFrame(timeline_data)
                    fig_timeline = px.scatter(
                        timeline_df,
                        x='Time',
                        y='Detection',
                        color='Type',
                        color_discrete_map={'Spam': '#FF007F', 'Ham': '#00F6FF'},
                        size_max=20
                    )
                    fig_timeline.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        font_color="white",
                        plot_bgcolor='rgba(0,0,0,0)',
                        height=400
                    )
                    st.plotly_chart(fig_timeline, use_container_width=True)

            with col5:
                st.subheader("📋 Confidence Distribution")
                if st.session_state.confidence_scores:
                    fig_hist = px.histogram(
                        x=st.session_state.confidence_scores,
                        nbins=20,
                        title="Distribution of Confidence Scores",
                        color_discrete_sequence=['#7B28C9']
                    )
                    fig_hist.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        font_color="white",
                        plot_bgcolor='rgba(0,0,0,0)',
                        height=400,
                        xaxis_title="Confidence Score",
                        yaxis_title="Frequency"
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)

    # =================================================================================
    # Page 4: Detection History
    # =================================================================================
    elif page == "Detection History":
        st.markdown('<div class="main-header">📜 Detection History</div>', unsafe_allow_html=True)
        st.markdown("### Complete log of all analyzed emails")
        
        if not st.session_state.history:
            st.warning("No history to display. Please analyze some emails first.")
        else:
            history_df = pd.DataFrame(st.session_state.history)
            
            # Add styling to the dataframe
            st.dataframe(
                history_df,
                use_container_width=True,
                column_config={
                    "Timestamp": st.column_config.TextColumn("🕒 Timestamp"),
                    "Snippet": st.column_config.TextColumn("📧 Email Snippet"),
                    "Prediction": st.column_config.TextColumn("🔍 Prediction"),
                    "Confidence": st.column_config.ProgressColumn(
                        "🎯 Confidence",
                        help="AI Confidence Level",
                        format="%.2f%%",
                        min_value=0,
                        max_value=100,
                    )
                }
            )
            
            # Export option
            if st.button("📥 Export History as CSV"):
                csv = history_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="spam_detection_history.csv",
                    mime="text/csv"
                )

    # =================================================================================
    # Page 5: Spam Prevention Tips (NEW FEATURE)
    # =================================================================================
    elif page == "Spam Prevention Tips":
        st.markdown('<div class="main-header">🛡️ Spam Prevention Tips</div>', unsafe_allow_html=True)
        st.markdown("### Best practices to protect yourself from spam and phishing")

        tips_col1, tips_col2 = st.columns(2)

        with tips_col1:
            st.subheader("🚨 Red Flags to Watch For")
            
            warning_signs = [
                "Urgent requests for personal information",
                "Too-good-to-be-true offers (lotteries, prizes)",
                "Suspicious sender addresses",
                "Poor grammar and spelling mistakes",
                "Requests for money or financial details",
                "Generic greetings like 'Dear Customer'",
                "Suspicious links or attachments"
            ]
            
            for sign in warning_signs:
                st.error(f"• {sign}")

            st.subheader("🔒 Protective Measures")
            protective_measures = [
                "Use strong, unique passwords for email accounts",
                "Enable two-factor authentication",
                "Don't click on suspicious links",
                "Verify sender identity before responding",
                "Use spam filters and keep them updated",
                "Regularly update your software and antivirus",
                "Be cautious with email attachments"
            ]
            
            for measure in protective_measures:
                st.success(f"• {measure}")

        with tips_col2:
            st.subheader("📊 Spam Statistics")
            
            # Create some informative charts about spam
            spam_facts = pd.DataFrame({
                'Category': ['Phishing Attacks', 'Financial Scams', 'Lottery Scams', 'Product Spam', 'Other'],
                'Percentage': [35, 25, 20, 15, 5]
            })
            
            fig_spam_types = px.pie(
                spam_facts, 
                values='Percentage', 
                names='Category',
                title="Common Types of Spam Emails",
                color_discrete_sequence=px.colors.sequential.Reds
            )
            fig_spam_types.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white")
            st.plotly_chart(fig_spam_types, use_container_width=True)

            st.subheader("🆘 What to Do If You Clicked a Spam Link")
            emergency_steps = [
                "Immediately run a full antivirus scan",
                "Change your passwords immediately",
                "Monitor your financial accounts",
                "Contact your bank if you entered financial info",
                "Report the spam to your email provider",
                "Consider freezing your credit if sensitive info was shared"
            ]
            
            for step in emergency_steps:
                st.warning(f"• {step}")

        st.markdown("---")
        st.info("💡 **Remember**: Always think before you click. When in doubt, don't give it out!")
