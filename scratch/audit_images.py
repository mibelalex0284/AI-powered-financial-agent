import sys
import os
sys.path.insert(0, os.path.abspath('.'))
import pandas as pd

events = pd.read_csv('dataset/financial_events.csv')
images = pd.read_csv('dataset/images.csv')

# Find all blank amounts
blank_events = events[events['amount'].isna()].copy()
print(f"Total blank events: {len(blank_events)}")

from code.forecasting import ImageAmountResolver

resolver = ImageAmountResolver(images)

audit_rows = []
for idx, row in blank_events.iterrows():
    e_id = str(row['event_id'])
    # Check link in images.csv
    img_match = images[images['related_event_id'] == e_id]
    if not img_match.empty:
        img_id = img_match.iloc[0]['image_id']
        img_file = f"dataset/media/images/{img_id}.png"
        img_exists = os.path.exists(img_file)
        img_desc = img_match.iloc[0].get('image_description', '')
    else:
        img_id = 'NONE'
        img_file = 'NONE'
        img_exists = False
        img_desc = 'No image linked'

    resolved_val = resolver.resolve_amount(e_id)
    audit_rows.append({
        'event_id': e_id,
        'user_id': row['user_id'],
        'category': row['category'],
        'currency': row['currency'],
        'image_id': img_id,
        'file_exists': img_exists,
        'resolved_amount': resolved_val,
        'description': row['description']
    })

df_audit = pd.DataFrame(audit_rows)
print(df_audit.to_string())
