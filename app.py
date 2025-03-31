import json
from datetime import datetime
import re
import pandas as pd
import json

def clean_products_file(file_path, output_path):
    # Clean products data by removing duplicates and standardizing columns.
    try:
        df = pd.read_json(file_path)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return pd.DataFrame()

    if df.empty:
        print(f"Warning: {file_path} is empty.")
        return df

    # Rename and standardize columns
    rename_dict = {
        'id': 'product_id', 'product_name': 'name', 'product_category': 'category',
        'cost': 'price', 'price_usd': 'price', 'MSRP': 'price', 'retail_price': 'price'
    }
    df = df.rename(columns=rename_dict)
    
    # Drop duplicate column names
    df = df.loc[:, ~df.columns.duplicated()]
    
    # Consolidate 'product_id'
    # if the product id exits and it is in double digit drop it
    product_id_cols = [col for col in df.columns if 'product_id' in col.lower()]
    if len(product_id_cols) > 1:
        df['product_id'] = df[product_id_cols].bfill(axis=1).iloc[:, 0]
        df = df.drop(columns=[col for col in product_id_cols if col != 'product_id'])

    # Consolidate 'price'
    price_cols = [col for col in df.columns if 'price' in col.lower()]
    if len(price_cols) > 1:
        df['price'] = df[price_cols].apply(lambda row: row.dropna().iloc[0] if not row.dropna().empty else None, axis=1)
        df = df.drop(columns=[col for col in price_cols if col != 'price'])

    # Ensure 'price' exists
    if 'price' not in df.columns:
        df['price'] = pd.NA

    # Convert 'price' to numeric
    df['price'] = pd.to_numeric(df['price'].astype(str).str.replace(r'[$,]', '', regex=True), errors='coerce')

    # Flatten 'details' column if it contains dictionaries
    if 'details' in df.columns and df['details'].apply(lambda x: isinstance(x, dict)).any():
        details_df = df['details'].apply(pd.Series)
        df = pd.concat([df.drop(columns=['details']), details_df], axis=1)

    # Convert list-like columns into strings to avoid unhashable type errors
    for col in df.columns:
        df[col] = df[col].apply(lambda x: json.dumps(x) if isinstance(x, (list, dict)) else x)

    # Keep only necessary columns
    keep_cols = ['product_id', 'name', 'category', 'price', 'features']
    df = df[[col for col in keep_cols if col in df.columns]].drop_duplicates()

    # Save cleaned data
    df.to_json(output_path, orient='records', indent=2)
    print(f"Cleaned products saved to {output_path}")
    return df

# Clean the products JSON file
products_clean = clean_products_file("products.json", "products_clean.json")



def clean_reviews_file(file_path, output_path):
    # Clean reviews data by standardizing columns and removing duplicates.
    # Load the CSV file
    try:
        df = pd.read_csv(file_path)
        print("Initial data loaded successfully.")
        print("Initial columns:", df.columns.tolist())
        print("Initial sample:\n", df.head())
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return pd.DataFrame()

    if df.empty:
        print(f"Warning: {file_path} is empty.")
        return df

    # Standardize column names (already consistent in this case, but included for robustness)
    df = df.rename(columns={
        'review': 'review_text'  # In case 'review' is used instead of 'review_text'
    })
    print("Columns after renaming:", df.columns.tolist())

    # Clean data
    # Convert 'review_date' to datetime
    if 'review_date' in df.columns:
        df['review_date'] = pd.to_datetime(df['review_date'], errors='coerce')
        print("Review dates converted to datetime.")

    # Convert 'rating' to nullable integer
    if 'rating' in df.columns:
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce').astype('Int64')  # Nullable integer
        print("Ratings converted to nullable integer.")

    # Clean 'review_text' (lowercase and strip whitespace)
    if 'review_text' in df.columns:
        df['review_text'] = df['review_text'].str.lower().str.strip()
        print("Review text cleaned (lowercase, stripped).")

    # Keep only necessary columns
    keep_cols = ['review_id', 'product_id', 'rating', 'review_text', 'review_date']
    df = df[[col for col in keep_cols if col in df.columns]]
    print("Columns retained:", df.columns.tolist())

    # Remove duplicate rows
    initial_rows = len(df)
    df = df.drop_duplicates()
    print(f"Removed {initial_rows - len(df)} duplicate rows. Final row count: {len(df)}")

    # Check for missing values
    print("Missing values in cleaned data:\n", df.isnull().sum())

    # Save cleaned data
    df.to_csv(output_path, index=False)
    print(f"Cleaned reviews saved to {output_path}")
    return df


def perform_eda(df):
    
    print("\n--- Exploratory Data Analysis ---")

    # 1. Rating Distribution
    print("\nRating Distribution:")
    print(df['rating'].value_counts().sort_index())

    # 2. Reviews per Product
    print("\nNumber of Reviews per Product:")
    reviews_per_product = df.groupby('product_id').size().sort_values(ascending=False)
    print(reviews_per_product)

    # 3. Average Rating per Product
    print("\nAverage Rating per Product:")
    avg_rating_per_product = df.groupby('product_id')['rating'].mean().sort_values()
    print(avg_rating_per_product)

    # 4. Review Trends Over Time
    print("\nReviews by Month:")
    df['month'] = df['review_date'].dt.to_period('M')
    monthly_reviews = df.groupby('month').size()
    print(monthly_reviews)

    # 5. Common Words in Reviews (basic keyword extraction)
    def extract_keywords(text):
        keywords = re.findall(r'\b(good|great|best|bad|okay|love|amazing|disappoint|quality|perfect)\b', text.lower())
        return keywords if keywords else None

    df['keywords'] = df['review_text'].apply(extract_keywords)
    keyword_counts = df['keywords'].explode().value_counts()
    print("\nCommon Keywords in Reviews:")
    print(keyword_counts)

# --- Main Execution ---

# File paths
input_file = "reviews.csv"
output_file = "reviews_clean.csv"

# Clean the reviews file
reviews_clean = clean_reviews_file(input_file, output_file)

# Perform EDA on cleaned data
if not reviews_clean.empty:
    perform_eda(reviews_clean)
else:
    print("No data available for EDA.")

print("\nData cleaning and analysis complete!")



def load_and_combine_json_files(file_paths):
    
    combined_data = []
    
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            combined_data.extend(data)
    
    return pd.DataFrame(combined_data)

def extract_message_info(messages):
    
    first_message = messages[0]
    last_message = messages[-1]
    
    # Extract key dates
    dates = [msg['date'] for msg in messages]
    try:
        start_date = datetime.strptime(dates[0], '%Y-%m-%d')
        end_date = datetime.strptime(dates[-1], '%Y-%m-%d')
        resolution_days = (end_date - start_date).days
    except:
        start_date = end_date = resolution_days = None
    
    # Extract sender domains
    from_domains = [msg['from'].split('@')[-1] for msg in messages]
    yourcompany_count = sum(1 for domain in from_domains if 'yourcompany.com' in domain)
    manufacturer_count = len(from_domains) - yourcompany_count
    
    # Extract key content features
    content_lengths = [len(msg['content']) for msg in messages]
    word_counts = [len(re.findall(r'\w+', msg['content'])) for msg in messages]
    
    # Check for attachments or evidence mentions
    has_evidence = any('attach' in msg['content'].lower() or 
                      'photo' in msg['content'].lower() or 
                      'evidence' in msg['content'].lower() 
                      for msg in messages)
    
    return {
        'num_messages': len(messages),
        'start_date': start_date,
        'end_date': end_date,
        'resolution_days': resolution_days,
        'yourcompany_msgs': yourcompany_count,
        'manufacturer_msgs': manufacturer_count,
        'avg_content_length': sum(content_lengths)/len(content_lengths),
        'avg_word_count': sum(word_counts)/len(word_counts),
        'has_evidence': has_evidence,
        'first_subject': first_message['subject'],
        'last_subject': last_message['subject'],
        'initial_response_days': (datetime.strptime(messages[1]['date'], '%Y-%m-%d') - 
                                 datetime.strptime(messages[0]['date'], '%Y-%m-%d')).days if len(messages) > 1 else None
    }

def clean_data(df):
    # Clean and transform the raw data
    # Extract message thread features
    message_info = df['messages'].apply(extract_message_info)
    message_info_df = pd.json_normalize(message_info)
    
    # Combine with original data
    cleaned_df = pd.concat([df.drop('messages', axis=1), message_info_df], axis=1)
    
    # Convert date columns to datetime
    date_cols = ['start_date', 'end_date']
    for col in date_cols:
        cleaned_df[col] = pd.to_datetime(cleaned_df[col])
    
    # Extract product category from product name (simple approach)
    cleaned_df['product_category'] = cleaned_df['product_name'].apply(
        lambda x: re.sub(r'[^a-zA-Z]', ' ', x).split()[0].lower() if pd.notnull(x) else None)
    
    # Clean resolution values
    cleaned_df['resolution'] = cleaned_df['resolution'].str.replace('_', ' ').str.title()
    
    # Calculate some additional metrics
    cleaned_df['msg_exchange_rate'] = cleaned_df['num_messages'] / cleaned_df['resolution_days']
    
    # Reorder columns
    cols = ['product_id', 'product_name', 'product_category', 'thread_id', 'issue', 
            'resolution', 'start_date', 'end_date', 'resolution_days', 'initial_response_days',
            'num_messages', 'msg_exchange_rate', 'yourcompany_msgs', 'manufacturer_msgs',
            'avg_content_length', 'avg_word_count', 'has_evidence', 'first_subject', 'last_subject']
    
    return cleaned_df[cols]

def save_cleaned_data(df, output_file):
    """Save cleaned data to CSV and JSON formats"""
    df.to_csv(output_file.replace('.json', '.csv'), index=False)
    df.to_json(output_file, orient='records', indent=2)
    print(f"Cleaned data saved to {output_file} and {output_file.replace('.json', '.csv')}")

def main():
    # File paths
    input_files = ['emails1.json', 'emails2.json']
    output_file = 'cleaned_email_data.json'
    
    # Load and combine data
    print("Loading and combining JSON files...")
    raw_df = load_and_combine_json_files(input_files)
    
    # Clean and transform data
    print("Cleaning and transforming data...")
    cleaned_df = clean_data(raw_df)
    
    # Save cleaned data
    save_cleaned_data(cleaned_df, output_file)
    
    print("Data cleaning complete!")


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from datetime import datetime
from wordcloud import WordCloud


df = pd.read_json('cleaned_email_data.json')

# Debug: Check initial date format
print("\nDebug - Date columns before conversion:")
print(f"start_date type: {type(df['start_date'].iloc[0])}")
print(f"end_date type: {type(df['end_date'].iloc[0])}")

# Convert date columns - handle multiple possible formats
try:
    # Try converting assuming ISO format strings
    df['start_date'] = pd.to_datetime(df['start_date'], format='%Y-%m-%d')
    df['end_date'] = pd.to_datetime(df['end_date'], format='%Y-%m-%d')
except:
    try:
        # Try converting Unix timestamps if string conversion fails
        df['start_date'] = pd.to_datetime(df['start_date'], unit='ms')
        df['end_date'] = pd.to_datetime(df['end_date'], unit='ms')
    except:
        # Fallback to generic conversion
        df['start_date'] = pd.to_datetime(df['start_date'])
        df['end_date'] = pd.to_datetime(df['end_date'])

# Debug: Check converted date format
print("\nDebug - Date columns after conversion:")
print(f"start_date type: {type(df['start_date'].iloc[0])}")
print(f"Sample start_date: {df['start_date'].iloc[0]}")

# Basic info
print(f"\nTotal threads: {len(df)}")
print(f"Time period: {df['start_date'].min().strftime('%Y-%m-%d')} to {df['end_date'].max().strftime('%Y-%m-%d')}")
print("\nResolution outcomes:")
print(df['resolution'].value_counts(normalize=True))

# Set style for plots - using available style
available_styles = plt.style.available
if 'seaborn' in available_styles:
    plt.style.use('seaborn')
elif 'ggplot' in available_styles:
    plt.style.use('ggplot')
else:
    plt.style.use('default')

sns.set_palette("pastel")

# 1. Temporal Analysis
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(df['resolution_days'], bins=8, kde=True)
plt.title('Distribution of Resolution Time (Days)')
plt.xlabel('Days to Resolve')

plt.subplot(1, 2, 2)
sns.boxplot(x='resolution', y='resolution_days', data=df)
plt.title('Resolution Time by Outcome')
plt.xlabel('Resolution Type')
plt.ylabel('Days to Resolve')
plt.tight_layout()
plt.show()

# 2. Communication Patterns
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
sns.scatterplot(x='num_messages', y='resolution_days', hue='resolution', data=df, s=100, ax=ax[0])
ax[0].set_title('Messages vs. Resolution Time')

sns.barplot(x='resolution', y='msg_exchange_rate', data=df, ax=ax[1])
ax[1].set_title('Message Exchange Rate by Outcome')
plt.tight_layout()
plt.show()

# 3. Evidence Impact
plt.figure(figsize=(8, 5))
evidence_impact = df.groupby('has_evidence')['resolution'].value_counts(normalize=True).unstack()
evidence_impact.plot(kind='bar', stacked=True)
plt.title('Resolution Outcomes by Evidence Presence')
plt.ylabel('Proportion of Cases')
plt.xticks([0, 1], ['No Evidence', 'Has Evidence'], rotation=0)
plt.legend(title='Resolution')
plt.show()

# 4. Product Analysis
plt.figure(figsize=(12, 6))
product_analysis = df.groupby('product_category')['resolution'].value_counts().unstack()
product_analysis.plot(kind='bar')
plt.title('Resolution Outcomes by Product Category')
plt.ylabel('Number of Cases')
plt.xlabel('Product Category')
plt.legend(title='Resolution')
plt.show()

# 5. Issue Word Cloud
plt.figure(figsize=(12, 6))
issue_terms = ' '.join(df['issue']).lower()
issue_terms = re.sub(r'[^a-zA-Z\s]', '', issue_terms)
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(issue_terms)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Most Common Terms in Issue Descriptions')
plt.show()

# 6. Response Time Analysis
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(df['initial_response_days'], bins=5)
plt.title('Initial Response Time Distribution')
plt.xlabel('Days to First Response')

plt.subplot(1, 2, 2)
sns.regplot(x='initial_response_days', y='resolution_days', 
           scatter_kws={'s': 100}, data=df)
plt.title('Initial vs Total Resolution Time')
plt.xlabel('Days to First Response')
plt.ylabel('Total Resolution Days')
plt.tight_layout()
plt.show()

print("\nKey Statistics:")
print(f"Average resolution time: {df['resolution_days'].mean():.1f} days")
print(f"Median resolution time: {df['resolution_days'].median()} days")
print(f"Average initial response: {df['initial_response_days'].mean():.1f} days")
print(f"Cases with evidence: {df['has_evidence'].sum()}/{len(df)}")


# sentiment analyser
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter

# Load data
df = pd.read_csv('reviews_clean.csv')

# Initialize VADER
sia = SentimentIntensityAnalyzer()

# Calculate sentiment scores
df['sentiment'] = df['review_text'].apply(lambda x: sia.polarity_scores(x)['compound'])

# Classify sentiment
df['sentiment_label'] = df['sentiment'].apply(lambda x: 'Positive' if x >= 0.6 
                                             else ('Negative' if x <= -0.6 
                                                   else 'Neutral'))

# Compare with ratings
sentiment_by_rating = df.groupby('rating')['sentiment'].mean()

# Product sentiment analysis
product_sentiment = df.groupby('product_id')['sentiment'].mean().sort_values(ascending=False)


# sentiment distribution
plt.figure(figsize=(8,5))
df['sentiment_label'].value_counts().plot(kind='bar')
plt.title('Review Sentiment Distribution')
plt.show()

# vs rating
plt.figure(figsize=(10,6))
sns.boxplot(x='rating', y='sentiment', data=df)
plt.title('Sentiment Scores by Star Rating')
plt.show()

# top positive and negative words
positive_words = ' '.join(df[df['sentiment_label']=='Positive']['review_text'])
wordcloud = WordCloud(width=800, height=400).generate(positive_words)
plt.imshow(wordcloud)
plt.axis('off')
plt.title('Most Common Positive Words')
plt.show()

if __name__ == "__main__":
    main()