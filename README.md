# 🛡️ AI-Powered Email Spam Detection System

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-FF6B6B?style=for-the-badge&logo=ai&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

**An intelligent spam detection system that protects your inbox using advanced AI algorithms**

*Developed during Machine Learning Internship at Oasis Infobyte*


[![LinkedIn](https://img.shields.io/badge/LinkedIn%20Post-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)]https://www.linkedin.com/posts/yash-gupta-115728294_machinelearning-ai-python-activity-7383460790298796032-5Dev?utm_source=share&utm_medium=member_desktop&rcm=ACoAAEdMbKEBUHYbmmhRUezMh7FBr-cIk1M-WtM

</div>

---

## 📊 Project Overview

This **Email Spam Detection System** leverages cutting-edge Machine Learning techniques to accurately classify emails as **spam** or **legitimate (ham)**. Built with a modern web interface, it provides real-time analysis with detailed insights and visual analytics.

### 🎯 Key Features

- 🔍 **Real-time Email Analysis** - Instant spam detection
- 📈 **Interactive Dashboard** - Beautiful visualizations and analytics  
- 🎯 **Confidence Scoring** - AI confidence levels for each prediction
- 📊 **Spam Indicator Detection** - Identifies suspicious patterns
- 💾 **File Upload Support** - Analyze .txt and .eml files
- 📱 **Responsive Design** - Modern, mobile-friendly interface
- 📜 **Session History** - Track all analyses in one place

---



## 🛠️ Technical Architecture

### 📋 Tech Stack

| Component | Technology |
|-----------|------------|
| **Frontend** | Streamlit |
| **Backend** | Python 3.8+ |
| **Machine Learning** | Scikit-learn, NLTK |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Plotly, Matplotlib |
| **Deployment** | Streamlit Cloud |

### 🔧 Machine Learning Pipeline
Raw Email Text
→ Text Preprocessing
→ TF-IDF Vectorization
→ ML Model Prediction
→ Confidence Scoring
→ Visual Results

text

---

## 📁 Project Structure
OIBSIP_ML_4/
│
├── app.py # Main Streamlit application
├── model.pkl # Trained ML model
├── tfidf_vectorizer.pkl # Fitted TF-IDF vectorizer
├── requirements.txt # Project dependencies
├── Email_Spam_Detection_Model_Training.ipynb # Model training notebook
├── assets/ # Images and screenshots
│ ├── banner.jpg
│ ├── workflow.png
│ └── results.png
└── README.md # Project documentation

text

---

## ⚡ Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/OIBSIP_ML_4.git
   cd OIBSIP_ML_4
Install dependencies

bash
pip install -r requirements.txt
Run the application

bash
streamlit run app.py
Access the app

Open your browser and go to: http://localhost:8501

🎮 How to Use
Navigate to "Real-Time Detector"

Choose input method:

📝 Paste email text directly

📁 Upload .txt or .eml file

Click "Analyze Email with AI"

View detailed results including:

Spam/Ham classification

Confidence score

Spam indicators detected

Text analysis metrics

🤖 Machine Learning Model
📊 Dataset
Source: Kaggle Spam Collection Dataset

Samples: 5,172 emails

Classes: Spam (747), Ham (4,825)

🔬 Model Training
Algorithms Tested: Naive Bayes, Logistic Regression, Random Forest

Feature Engineering: TF-IDF with 5,000 features

Text Preprocessing: Lowercasing, stopword removal, stemming

Best Model: Logistic Regression (Highest Accuracy)

📈 Performance Metrics
Accuracy: 98.5%

Precision: 97.8%

Recall: 96.2%

F1-Score: 97.0%

🎨 Features in Detail
1. Real-Time Detection
Instant analysis with < 2 second response time

Live confidence gauge visualization

Detailed spam indicator breakdown

2. Analytics Dashboard
Interactive pie charts and bar graphs

Confidence trend analysis

Spam word cloud visualization

Real-time metrics tracking

3. Advanced Analytics
Detection timeline

Confidence distribution

Performance metrics

Session statistics

4. User Experience
Modern dark theme with neon accents

Glassmorphism design elements

Smooth animations and transitions

Mobile-responsive layout


🎓 Learning Outcomes
Technical Skills Enhanced
✅ End-to-end Machine Learning project development

✅ Natural Language Processing (NLP) techniques

✅ Model deployment and web integration

✅ Data visualization and dashboard creation

✅ Professional code structure and documentation

Professional Growth
✅ Project management and planning

✅ Problem-solving and debugging

✅ User interface design principles

✅ Version control with Git/GitHub

✅ Professional documentation writing

🔮 Future Enhancements
Deep Learning model integration (BERT, Transformers)

Multi-language support

Email attachment analysis

Real-time email integration (Gmail API)

Advanced phishing detection

Mobile application development

👨‍💻 Developer
YASH GUPTA
Machine Learning Intern at Oasis Infobyte



🙏 Acknowledgments
Oasis Infobyte for providing this internship opportunity

Mentors and guides for their valuable support

Open-source community for amazing libraries and tools

<div align="center">
⭐ If this project helped you, don't forget to give it a star!
Built with ❤️ as part of Oasis Infobyte Machine Learning Internship



</div> ```

