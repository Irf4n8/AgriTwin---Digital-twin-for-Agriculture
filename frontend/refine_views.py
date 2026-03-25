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
        
    # 1. Smooth transitions on the body and beautiful text-selection
    content = re.sub(
        r'<body class="([^"]*?)bg-gray-50([^"]*?)"',
        r'<body class="\1bg-slate-50 dark:bg-slate-900 transition-colors duration-500 selection:bg-green-500 selection:text-white\2"',
        content
    )

    # 2. Convert standard gray text to more premium Slate and handle dark mode directly at utility level
    # Only replace whole words to avoid messing with border-gray-500 etc
    content = re.sub(r'\btext-gray-800\b', r'text-slate-800 dark:text-slate-100', content)
    content = re.sub(r'\btext-gray-500\b', r'text-slate-500 dark:text-slate-400', content)
    content = re.sub(r'\btext-gray-600\b', r'text-slate-600 dark:text-slate-300', content)
    content = re.sub(r'\border-gray-200\b', r'border-slate-200 dark:border-slate-700/50', content)
    content = re.sub(r'\bbg-gray-50\b', r'bg-slate-50 dark:bg-slate-900/50', content)
    
    # 3. Add soft dropshadow to major SVGs and icons
    def text_4xl_icon_replacer(match):
        inner = match.group(0)
        if 'drop-shadow' not in inner:
            inner = inner[:-1] + ' drop-shadow-md"'
        return inner

    content = re.sub(r'class="[^"]*?text-4xl[^"]*?mb-3[^"]*?"', text_4xl_icon_replacer, content)

    with open(path, 'w', encoding='utf-8') as file:
        file.write(content)
        
    print(f"Refined view styling in {f}")
