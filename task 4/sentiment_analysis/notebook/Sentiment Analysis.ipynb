# -*- coding: utf-8 -*-
"""
Created on Sat Nov 15 06:32:07 2025

@author: EBUNOLUWASIMI
"""
from utf8_converter import convert_to_utf8 as uc
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from transformers import pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
nltk.download('stopwords')
nltk.download('vader_lexicon')

#Ensure proper encoding
input_file = r"C:\Users\EBUNOLUWASIMI\Dropbox\GM\Data Analytics\prodigy\task 4\twitter_training.csv"
file = uc(input_file, output_file=None)

# Create sample data
data = pd.read_csv(file,parse_dates=True)
df = pd.DataFrame(data)
print(df.head().to_string())

# Clean text
def clean_text(text):
    if not isinstance(text, str):
        text = str(text)
    if pd.isna(text):
        return ""
    text = re.sub(r'http\S+|@\S+|#\S+', '', text)  # remove links, mentions, hashtags
    text = re.sub(r'[^A-Za-z\s]', '', text)        # remove punctuation
    text = text.lower()                            # lowercase
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])
    return text

df.iloc[:,3] = df.iloc[:,3].fillna("")
df['text'] = df.iloc[:,3].apply(clean_text)
print(df)

print("Data cleaning completed! Now proceeding to Sentiment Analysis >>>")
print("Two models are available: Press v for Vadar_Lexicon model or T for Transformer model >>>")
response = input().lower()

if response in ("V", "v"):
    print("You have chosen to use Vadar_Lexicon model. Now proceeding with Sentiment Analysis \
          using Vadar_Lexicon")
    # Sentiment Analysis
    sia = SentimentIntensityAnalyzer()
    df['Sentiment_Score'] = df['text'].apply(lambda x: sia.polarity_scores(x)['compound'])

    def label_sentiment(score):
        if score > 0.05:
            return 'Positive'
        elif score < -0.05:
            return 'Negative'
        else:
            return 'Neutral'

    df['Sentiment'] = df['Sentiment_Score'].apply(label_sentiment)
    analysis = df[['text', 'Sentiment_Score', 'Sentiment']]
    print(df[['text', 'Sentiment_Score', 'Sentiment']])
    analysis.to_csv("Vadar_Sentiment_Analysis.txt", sep="\t", index=False)
    

elif response in ("T","t"):
    print("You have chosen to use Transformers model. Now proceeding with Sentiment Analysis \
          using Transformers")
    classifier = pipeline("sentiment-analysis")
    df['Sentiment'] = df['text'].apply(lambda x: classifier(x)[0]['label'])

#Sentiment Distribution
sns.countplot(x='Sentiment', data=df, palette='coolwarm')
plt.title('Overall Sentiment Distribution')
plt.show()

#Word Cloud foreach sentiment
positive_text = ' '.join(df[df['Sentiment']=='Positive']['text'])
negative_text = ' '.join(df[df['Sentiment']=='Negative']['text'])

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.imshow(WordCloud(colormap='Greens').generate(positive_text))
plt.axis('off')
plt.grid(True)
plt.title('Positive Words')

plt.subplot(1,2,2)
plt.imshow(WordCloud(colormap='Reds').generate(negative_text))
plt.axis('off')
plt.title('Negative Words')
plt.grid(True)
plt.show()

#Trends over time
df['Date'] = pd.to_datetime(df.iloc[:,0],dayfirst=True, errors='coerce')
df = df.dropna(subset=['Date'])
df = df.sort_values('Date')
df['Sentiment'] = df['Sentiment'].astype(str)
sentiment_trendD = df.groupby(['Date', 'Sentiment']).size().unstack(fill_value=0)
sentiment_trendD = sentiment_trendD.sort_index()
sentiment_trendD.plot(kind='line', marker='o')
plt.figure(figsize=(12,6))
sentiment_trendD.plot(kind='line', marker='o', ax=plt.gca())
plt.title('Daily Sentiment Trend')
plt.ylabel('Number of Mentions')
plt.xlabel('Date')
plt.grid(True)
plt.legend(title='Sentiment')
plt.show()

monthly_trend = (
    df.set_index('Date')
      .resample('M')['Sentiment']
      .value_counts()
      .unstack(fill_value=0)
)

# Plot monthly trend
plt.figure(figsize=(12,6))
monthly_trend.plot(kind='line', marker='o', ax=plt.gca())
plt.title('Monthly Sentiment Trend')
plt.ylabel('Number of Mentions')
plt.xlabel('Month')
plt.grid(True)
plt.legend(title='Sentiment')
plt.show()


