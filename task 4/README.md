# 🧠 Sentiment Analysis on 75K+ Online Comments  
### NLP • Data Analytics • Visualization • VADER/TextBlob • Word Cloud • Time-Series Insight

This project performs a full sentiment analysis workflow on over **75,000 user comments**, uncovering emotional patterns, key themes, and monthly sentiment trends. It is designed for real-world applications such as brand monitoring, customer experience analytics, and social behavior research.

---

## 📊 Key Results

### **1. Overall Sentiment Distribution**
Positive sentiments lead the conversation, followed by a substantial volume of negative reactions. Neutral comments are the least common — typical of emotionally expressive social platforms.

### **2. Word Cloud Insights**
- **Positive Words:** *game, new, thank, love, good*  
- **Negative Words:** *shit, problem, dead, fuck, bad*  

These words highlight both user excitement and major points of frustration.

### **3. Monthly Sentiment Trend**
A highly volatile trend with spikes influenced by:
- Product updates  
- Outages  
- New releases  
- Social events  

The chart is ideal for seasonality analysis and forecasting.

---

## 🧰 Technologies Used

- **Python**
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - wordcloud
  - nltk / VADER / TextBlob
- **Jupyter Notebook**
- **Data Visualization**
- **Time-Series Analysis**
- **NLP Preprocessing**

---

## 📂 Project Structure

sentiment-analysis/

├── data/

     ├── comments.csv
     
     └── cleaned_comments.csv

├── notebooks/

     └── sentiment_analysis.ipynb

├── outputs/

     ├── sentiment_distribution.png
     
     ├── wordcloud_positive.png
     
     ├── wordcloud_negative.png
     
     └── monthly_sentiment_trend.png

├── README.md

└── requirements.txt

````

## 🚀 How to Run the Project

```bash
git clone https://github.com/<your-username>/sentiment-analysis.git
cd sentiment-analysis
pip install -r requirements.txt
jupyter notebook
````

Open `sentiment_analysis.ipynb` and run all cells.

---

## 📌 Future Enhancements

* Build a Streamlit dashboard
* Add emotion classification (anger, joy, sadness, etc.)
* Deploy as an API for real-time sentiment tracking
* Add LLM-based sentiment interpretation

---

## 📬 Contact

**Opeyemi Sadiku**
Data Analyst • NLP • Machine Learning
LinkedIn: [https://linkedin.com/in/opeyemi-sadiku](https://linkedin.com/in/opeyemi-sadiku)

If this project helps you, please ⭐ star the repo!

