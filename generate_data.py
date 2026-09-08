import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Load existing seed data
seed_df = pd.read_csv("backend/market.csv")

# We will generate data for the last 30 days based on these seed rows
new_rows = []

end_date = datetime(2026, 1, 2) # The date in the current file
start_date = end_date - timedelta(days=30)

for index, row in seed_df.iterrows():
    base_min = row['Min_Price']
    base_max = row['Max_Price']
    base_modal = row['Modal_Price']
    
    # Generate 30 days of history
    current_date = start_date
    while current_date <= end_date:
        # Random fluctuation +/- 10%
        fluctuation = 1 + random.uniform(-0.1, 0.1)
        
        # Ensure prices make sense (Min < Modal < Max)
        new_modal = int(base_modal * fluctuation)
        new_min = int(new_modal * 0.9)
        new_max = int(new_modal * 1.1)
        
        # Create new entry
        new_entry = row.copy()
        new_entry['Arrival_Date'] = current_date.strftime("%d-%m-%Y")
        new_entry['Min_Price'] = new_min
        new_entry['Max_Price'] = new_max
        new_entry['Modal_Price'] = new_modal
        
        new_rows.append(new_entry)
        
        current_date += timedelta(days=1)

# Create enhanced dataframe
enhanced_df = pd.DataFrame(new_rows)

# Save back to csv
enhanced_df.to_csv("backend/market.csv", index=False)
print(f"Generated {len(enhanced_df)} rows of data.")
