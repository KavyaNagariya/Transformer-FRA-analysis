⚡ AI-Based Intelligent FRA Diagnostic System
🚀 Overview

The AI-Based Intelligent Diagnostic System for Frequency Response Analysis (FRA) is an advanced platform designed to automate the analysis of transformer FRA data using machine learning, anomaly detection, and expert system logic.

This system eliminates the dependency on manual interpretation by experts and provides accurate, consistent, and real-time fault diagnosis with a professional interactive dashboard.

🎯 Problem Statement

Frequency Response Analysis (FRA) is widely used to detect internal faults in power transformers such as:

Winding deformation
Core displacement
Insulation degradation

However:

Requires expert interpretation
Complex signal patterns
Multi-vendor incompatible data formats
Slow and inconsistent diagnostics
💡 Solution

This project introduces a fully automated AI-driven FRA diagnostic platform that:

Parses multi-format FRA data
Extracts meaningful signal features
Applies machine learning for fault classification
Uses anomaly detection for unseen patterns
Integrates expert system logic for recommendations
Visualizes results through an interactive dashboard
Generates professional diagnostic reports
🧠 Key Features
🔄 Universal Data Parser
Supports CSV and Excel FRA data
Auto-detects format and standardizes input
📊 Advanced Signal Analysis
Peak detection
Frequency band energy analysis
Statistical signal features
🤖 Machine Learning Model
Fault classification using Random Forest
Predicts:
Healthy
Winding deformation
Insulation issues
Core displacement
🚨 Anomaly Detection
Isolation Forest for detecting unknown faults
Identifies abnormal FRA patterns
🧠 Expert System
Rule-based interpretation of signals
Provides:
Fault type
Severity level
Maintenance recommendation
📈 Interactive Visualization Dashboard
FRA curve (log scale)
Comparison graph (healthy vs faulty)
Difference plot (deviation analysis)
Feature insights panel
Confidence & severity indicators
🌙 Modern Dark-Themed UI
Professional dashboard design
Interactive graphs (Plotly/Chart.js)
Smooth animations & clean layout
📄 Report Generation
Auto-generated PDF diagnostic report
Includes graphs, results, and recommendations
🏗️ System Architecture
FRA Data Input
      ↓
Universal Parser
      ↓
Feature Extraction
      ↓
ML Model + Anomaly Detection
      ↓
Expert System
      ↓
Visualization Dashboard
      ↓
PDF Report
📂 Project Structure
FRA_AI_Data/
│
├── app/                    # Flask application
├── src/
│   ├── parser/             # Data parsing modules
│   ├── features/           # Feature extraction
│   ├── models/             # ML & anomaly detection
│   ├── expert/             # Rule-based system
│   ├── utils/              # Plotting & report generation
│
├── static/                 # CSS, JS
├── templates/              # HTML files
├── data/                   # FRA datasets
├── requirements.txt
└── README.md
⚙️ Installation & Setup
1️⃣ Clone Repository
git clone <your-repo-link>
cd FRA_AI_Data
2️⃣ Create Virtual Environment
python -m venv venv
venv\Scripts\activate   # Windows
3️⃣ Install Dependencies
pip install -r requirements.txt
4️⃣ Run Application
python app/app.py
🌐 Open in Browser
http://127.0.0.1:5000/
🧪 Usage
Upload FRA file (CSV/Excel)
System processes data automatically
View:
FRA graphs
Fault classification
Confidence score
Feature insights
Recommendations
Download diagnostic report
📊 Sample Output
{
  "fault": "Winding Deformation",
  "confidence": 87,
  "severity": "High",
  "anomaly_score": -0.42,
  "recommendation": "Immediate inspection required"
}
🚀 Future Improvements
Real-world FRA dataset integration
Deep learning (CNN-based signal analysis)
Multi-file comparison
Cloud deployment
Transformer health monitoring dashboard
🏆 Hackathon Value

This project stands out due to:

Real-world industrial application
AI + Electrical Engineering integration
End-to-end automation
Professional UI/UX
Scalable architecture
👨‍💻 Tech Stack
Backend: Python, Flask
ML: Scikit-learn
Signal Processing: NumPy, SciPy
Visualization: Matplotlib / Plotly
Frontend: HTML, CSS, JavaScript
Reports: ReportLab
🤝 Contribution

Contributions are welcome!
Feel free to fork, improve, and submit pull requests.

📜 License

This project is for academic and research purposes.

💬 Final Note

This project bridges the gap between traditional transformer diagnostics and modern AI-driven predictive maintenance, making FRA analysis faster, smarter, and more reliable.