import os
import re

files = ['index.html', 'market.html', 'live_prices.html', 'map.html', 'simulation.html', 'yield.html', 'profit_calculator.html']
base_dir = r'e:\Agri-Twin Digital Twin For Agriculture\frontend'

for f in files:
    path = os.path.join(base_dir, f)
    if not os.path.exists(path):
        continue
    with open(path, 'r', encoding='utf-8') as file:
        content = file.read()
        
    # 1. Enhance primary buttons
    content = re.sub(
        r'bg-green-600\s+hover:bg-green-700', 
        r'bg-gradient-to-r from-green-500 to-emerald-600 hover:from-green-600 hover:to-emerald-700 hover-lift shadow-sm hover:shadow-md border border-transparent', 
        content
    )
    
    # 2. Enhance white/secondary action buttons
    content = re.sub(
        r'bg-white\s+dark:bg-gray-800\s+border\s+border-gray-200',
        r'bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 hover-lift',
        content
    )
    
    # 3. Add hover-lift and larger rounding to standard white cards
    # We only want to replace if hover-lift isn't already there.
    def card_replacer(match):
        inner = match.group(0)
        if 'hover-lift' not in inner:
            inner = inner.replace('rounded-xl', 'rounded-2xl hover-lift')
            inner = inner.replace('rounded-lg', 'rounded-xl hover-lift') # maybe also bump lg to xl
        return inner

    # match anything like class=" ... bg-white ... rounded-xl ... "
    content = re.sub(r'class="[^"]*?bg-white[^"]*?(?:rounded-xl|rounded-lg)[^"]*?"', card_replacer, content)

    # 4. Make all large headers use Outfit font and tight tracking
    def header_replacer(match):
        inner = match.group(0)
        if 'outfit' not in inner:
            # inject outfit before the closing quote
            inner = inner[:-1] + ' outfit tracking-tight"' 
        return inner

    content = re.sub(r'class="[^"]*?text-[34]xl[^"]*?"', header_replacer, content)

    # 5. Fix body background dynamically to use light/dark rules properly without inline
    # Mostly handled by styles.css but removing hardcoded bg-gray-50 from body can be good, 
    # though bg-gray-50 dark:bg-slate-900 is fine. style.css overrides.

    with open(path, 'w', encoding='utf-8') as file:
        file.write(content)
        
    print(f"Enhanced components in {f}")
