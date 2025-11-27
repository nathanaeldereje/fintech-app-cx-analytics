# Ethiopian Bank App Reviews – Customer Experience Analytics

**Omega Consultancy – Week 2 Data Engineering & Analytics Challenge**  
Analyzing user satisfaction for the three largest Ethiopian mobile banking apps:
- Commercial Bank of Ethiopia (CBE Birr)
- Bank of Abyssinia (Abyssinia Mobile)
- Dashen Bank (Amole)

## Business Objective
Help Ethiopian banks improve customer retention and satisfaction by turning raw Google Play Store reviews into actionable insights on sentiment, recurring themes, pain points, and feature requests.

## Key Features
- Scraped 1,200+ real user reviews using `google-play-scraper`
- Sentiment analysis with DistilBERT (and VADER comparison)
- Keyword & n-gram extraction + manual + rule-based theme clustering
- Cleaned data stored in PostgreSQL with proper relational schema
- Insightful visualizations (Matplotlib/Seaborn) + stakeholder-ready recommendations

## Project Structure

Ethiopian-Bank-CX-Analysis/
├── data/                  # Raw & processed CSV files
├── scripts/               # Production-ready Python scripts
│   ├── scrape_reviews.py
│   ├── preprocess_reviews.py
│   ├── sentiment_analysis.py
│   ├── theme_extraction.py
│   └── load_to_postgres.py
├── notebooks/             # Exploratory analysis & visualization
│   ├── 01_eda.ipynb
│   ├── 02_sentiment_analysis.ipynb
│   ├── 03_theme_clustering.ipynb
│   └── 04_final_insights.ipynb
├── sql/                   # Schema + sample queries
├── .gitignore
├── requirements.txt
└── README.md



## Quick Start

```bash
git clone https://github.com/yourname/ethiopian-bank-app-reviews.git
cd ethiopian-bank-app-reviews
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python scripts/scrape_reviews.py   # collects ~400+ reviews per bank
```


## Banks Analyzed
Bank,App Name,Play Store Rating
Commercial Bank of Ethiopia,CBE Birr,4.2★
Bank of Abyssinia,Abyssinia Mobile Banking,3.4★
Dashen Bank,Amole,4.1★

Final report (PDF) and all visualizations available in /notebooks/04_final_insights.ipynb and the submitted report.

Challenge completed – Dec 2025
Built by Nathanael Dereje