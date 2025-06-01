import json
from datetime import datetime

# Load the JSON file
input_file = "Raw Data v4.json"
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Step 1: Filter articles from 2023 and 2024
filtered_data = []
for article in data:
    try:
        # Parse the published date
        pub_date = datetime.strptime(article['published_datetime_utc'], '%Y-%m-%dT%H:%M:%S.%fZ')
        year = pub_date.year
        # Keep articles from 2023 and 2024
        if year in [2023, 2024]:
            filtered_data.append(article)
    except (ValueError, KeyError):
        # Skip articles with invalid or missing dates
        continue

# Step 2: Remove articles with null Full_Article
filtered_data = [article for article in filtered_data if article.get('Full_Article') is not None]

# Step 3: Remove duplicates based on the 'link' field
seen_links = set()
unique_data = []
for article in filtered_data:
    link = article.get('link')
    if link and link not in seen_links:
        seen_links.add(link)
        unique_data.append(article)

# Step 4: Save the filtered data to a new JSON file
output_file = "Filtered_Raw_Data_v4.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(unique_data, f, indent=4, ensure_ascii=False)

print(f"Filtered data saved to {output_file}. Total articles: {len(unique_data)}")