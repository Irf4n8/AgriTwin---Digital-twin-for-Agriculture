import requests
from bs4 import BeautifulSoup
import json

URLS = {
    "coimbatore": "https://vegetablemarketprice.com/market/coimbatore/today",
    "koyambedu": "https://vegetablemarketprice.com/market/koyambedu/today",
    "chennai": "https://vegetablemarketprice.com/market/chennai/today",
    "bangalore": "https://vegetablemarketprice.com/market/bangalore/today"
}

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def scrape_market(market_key="coimbatore"):
    url = URLS.get(market_key, URLS["coimbatore"])
    resp = requests.get(url, headers=headers, timeout=10)
    if resp.status_code != 200:
        print(f"Failed to fetch {url}, status: {resp.status_code}")
        return []

    soup = BeautifulSoup(resp.text, 'html.parser')
    rows = soup.find_all('tr', class_='todayVegetableTableRows')
    data = []
    for row in rows:
        tds = row.find_all('td')
        if len(tds) >= 5:
            img_tag = tds[0].find('img')
            img_url = ""
            if img_tag and img_tag.get('src'):
                src = img_tag['src']
                if src.startswith('/'):
                    img_url = "https://vegetablemarketprice.com" + src
                else:
                    img_url = src
            
            veg_name = tds[1].get_text(strip=True)
            wholesale_price = tds[2].get_text(strip=True)
            retail_price = tds[3].get_text(strip=True)
            unit = tds[4].get_text(strip=True)

            data.append({
                "image": img_url,
                "name": veg_name,
                "wholesale_price": wholesale_price,
                "retail_price": retail_price,
                "unit": unit
            })
    return data

if __name__ == "__main__":
    result = scrape_market("coimbatore")
    print(f"Scraped {len(result)} vegetables from Coimbatore:")
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    print(json.dumps(result[:5], indent=2, ensure_ascii=False))
