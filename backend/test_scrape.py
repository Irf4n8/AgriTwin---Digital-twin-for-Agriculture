import requests
from bs4 import BeautifulSoup

url = "https://vegetablemarketprice.com/market/tamilnadu/today"
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"}
try:
    r = requests.get(url, headers=headers)
    print(f"Status: {r.status_code}, Length: {len(r.text)}")
    soup = BeautifulSoup(r.text, 'html.parser')
    tables = soup.find_all('table')
    print(f"Tables found: {len(tables)}")
    if tables:
        print("First table rows:")
        rows = tables[0].find_all('tr')
        with open('scrape_out.txt', 'w', encoding='utf-8') as f:
            for row in rows[1:5]:
                cols = [col.text.strip() for col in row.find_all(['td', 'th'])]
                f.write(str(cols) + '\n')
    else:
        print("No tables found. Looking into scripts...")
        scripts = soup.find_all('script')
        for i, s in enumerate(scripts):
            if s.string and 'data' in s.string.lower():
                print(f"Script {i} length: {len(s.string)}")
                print(s.string[:200])
except Exception as e:
    print("Error:", e)
