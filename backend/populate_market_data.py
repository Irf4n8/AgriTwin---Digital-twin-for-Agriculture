import sqlite3
import random
from datetime import datetime, timedelta
import os

# Database Path
DB_NAME = "agritwin.db"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, DB_NAME)

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# Sample Data Config
STATES = {
    "Tamil Nadu": ["Coimbatore", "Madurai", "Salem", "Chennai", "Erode"],
    "Punjab": ["Ludhiana", "Amritsar", "Bathinda", "Patiala", "Jalandhar"],
    "Maharashtra": ["Pune", "Nashik", "Nagpur", "Mumbai", "Aurangabad"],
    "Rajasthan": ["Jaipur", "Kota", "Jodhpur", "Udaipur", "Bikaner"],
    "Karnataka": ["Bangalore", "Mysore", "Hubli", "Mangalore", "Belgaum"],
    "Uttar Pradesh": ["Lucknow", "Kanpur", "Varanasi", "Agra", "Meerut"],
    "Madhya Pradesh": ["Bhopal", "Indore", "Gwalior", "Jabalpur", "Ujjain"],
    "Gujarat": ["Ahmedabad", "Surat", "Vadodara", "Rajkot", "Bhavnagar"],
    "Haryana": ["Faridabad", "Gurgaon", "Panipat", "Ambala", "Yamunanagar"],
    "West Bengal": ["Kolkata", "Howrah", "Durgapur", "Asansol", "Siliguri"],
    "Bihar": ["Patna", "Gaya", "Bhagalpur", "Muzaffarpur", "Purnia"],
    "Andhra Pradesh": ["Visakhapatnam", "Vijayawada", "Guntur", "Nellore", "Kurnool"],
    "Telangana": ["Hyderabad", "Warangal", "Nizamabad", "Karimnagar", "Khammam"],
    "Kerala": ["Thiruvananthapuram", "Kochi", "Kozhikode", "Thrissur", "Kollam"],
    "Assam": ["Guwahati", "Silchar", "Dibrugarh", "Jorhat", "Nagaon"],
    "Odisha": ["Bhubaneswar", "Cuttack", "Rourkela", "Berhampur", "Sambalpur"],
    "Chhattisgarh": ["Raipur", "Bhilai", "Bilaspur", "Korba", "Durg"],
    "Jharkhand": ["Ranchi", "Jamshedpur", "Dhanbad", "Bokaro", "Hazaribagh"],
    "Uttarakhand": ["Dehradun", "Haridwar", "Roorkee", "Haldwani", "Rudrapur"],
    "Himachal Pradesh": ["Shimla", "Mandi", "Dharamshala", "Solan", "Kullu"],
    "Jammu & Kashmir": ["Srinagar", "Jammu", "Anantnag", "Baramulla", "Kathua"],
    "Goa": ["Panaji", "Margao", "Vasco da Gama", "Mapusa", "Ponda"],
    "Tripura": ["Agartala", "Udaipur", "Dharmanagar", "Kailasahar", "Ambassa"],
    "Manipur": ["Imphal", "Thoubal", "Bishnupur", "Churachandpur", "Ukhrul"],
    "Meghalaya": ["Shillong", "Tura", "Jowai", "Nongpoh", "Williamnagar"],
    "Nagaland": ["Kohima", "Dimapur", "Mokokchung", "Tuensang", "Wokha"],
    "Arunachal Pradesh": ["Itanagar", "Naharlagun", "Pasighat", "Tawang", "Ziro"],
    "Mizoram": ["Aizawl", "Lunglei", "Champhai", "Serchhip", "Kolasib"],
    "Sikkim": ["Gangtok", "Namchi", "Gyalshing", "Mangan", "Singtam"],
    "Delhi": ["New Delhi", "North Delhi", "South Delhi", "East Delhi", "West Delhi"],
    "Chandigarh": ["Chandigarh"],
    "Puducherry": ["Puducherry", "Karaikal", "Mahe", "Yanam"]
}

COMMODITIES = {
    "Rice": (2500, 4500),
    "Wheat": (2000, 3500),
    "Tomato": (1500, 8000),      # High variance
    "Potato": (1000, 3000),
    "Onion": (1500, 9000),       # High variance
    "Cotton": (5000, 8000),
    "Maize": (1800, 2600),
    "Apple": (6000, 15000),
    "Banana": (2000, 4000),
    "Soybean": (3500, 5500),
    "Mustard": (4000, 6000),
    "Sugarcane": (250, 400),     # Per quintal usually low but let's assume standard unit
    "Turmeric": (6000, 10000),
    "Chilly": (8000, 18000)
}

VARIETIES = ["Common", "Grade A", "Hybrid", "Desi", "Organic"]
GRADES = ["FAQ", "Medium", "Large", "Small", "Premium"]

def generate_market_data(num_records=5000):
    print(f"Generating {num_records} market records...")
    
    data = []
    
    # Date range: 2023 to 2026
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2026, 12, 31)
    days_range = (end_date - start_date).days
    
    for _ in range(num_records):
        state = random.choice(list(STATES.keys()))
        district = random.choice(STATES[state])
        market = f"{district} Mandi"
        
        commodity = random.choice(list(COMMODITIES.keys()))
        min_p_range, max_p_range = COMMODITIES[commodity]
        
        # Randomize price within range with some volatility based on "season" (randomized here)
        base_price = random.uniform(min_p_range, max_p_range)
        volatility = random.uniform(-0.2, 0.2)
        
        modal_price = base_price * (1 + volatility)
        min_price = modal_price * random.uniform(0.85, 0.95)
        max_price = modal_price * random.uniform(1.05, 1.15)
        
        # Date
        random_days = random.randint(0, days_range)
        arrival_date = (start_date + timedelta(days=random_days)).strftime("%Y-%m-%d")
        
        data.append((
            state, district, market, commodity, 
            random.choice(VARIETIES), 
            random.choice(GRADES), 
            arrival_date, 
            round(min_price, 2), 
            round(max_price, 2), 
            round(modal_price, 2)
        ))
        
    return data

def populate_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # OPTIONAL: Clear existing data to avoid duplicates or messy mix
    # user asked to "make it workable", meaningful data is better than random mix
    print("Clearing existing market_data...")
    cursor.execute("DELETE FROM market_data")
    
    rows = generate_market_data(3000) # 3000 records approx
    
    print("Inserting data into DB...")
    cursor.executemany('''
        INSERT INTO market_data (state, district, market, commodity, variety, grade, arrival_date, min_price, max_price, modal_price)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', rows)
    
    conn.commit()
    count = cursor.execute("SELECT COUNT(*) FROM market_data").fetchone()[0]
    conn.close()
    print(f"Success! Total records in market_data: {count}")

if __name__ == "__main__":
    populate_db()
