import os
import re

files = ['index.html', 'market.html', 'live_prices.html', 'map.html', 'simulation.html', 'yield.html', 'profit_calculator.html']
base_dir = r"e:\Agri-Twin Digital Twin For Agriculture\frontend"

for f in files:
    path = os.path.join(base_dir, f)
    if not os.path.exists(path):
        continue
    
    with open(path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Remove all html.dark {...} overrides present in inline tags
    content = re.sub(r'html\.dark[^{]*\{[^}]*\}', '', content)
    
    # Remove .glass {...} block
    content = re.sub(r'\.glass\s*\{[^}]*\}', '', content)
    
    # Remove body {...} block which just has the basic font overrides that styles.css handles
    content = re.sub(r'(?s)body\s*\{[^}]*font-family:[^}]*\}', '', content)

    # Clean up any leftover comments like /* Dark Mode Manual Overrides */ inside <style>
    content = re.sub(r'/\*.*?\*/', '', content)

    # Clean up entirely empty <style> </style> blocks
    content = re.sub(r'<style>\s*</style>', '', content)

    with open(path, 'w', encoding='utf-8') as file:
        file.write(content)
        
    print(f"Cleaned inline conflicting styles from {f}")
