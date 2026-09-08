import os
import re

files = ['index.html', 'market.html', 'live_prices.html', 'map.html', 'simulation.html', 'yield.html', 'profit_calculator.html']
base_dir = r'e:\Agri-Twin Digital Twin For Agriculture\frontend'

nav_links_data = [
    ('index.html', 'Dashboard'),
    ('market.html', 'Market'),
    ('live_prices.html', 'Live Prices'),
    ('map.html', 'Soil Map'),
    ('simulation.html', 'Simulation'),
    ('yield.html', 'Yield Plan'),
    ('profit_calculator.html', 'Profit Calculator')
]

for f in files:
    path = os.path.join(base_dir, f)
    with open(path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Generate the links HTML
    links_html = ""
    for link_file, name in nav_links_data:
        if link_file == f:
            classes = "bg-white bg-opacity-20 px-4 py-2 rounded-lg text-sm font-bold shadow-sm whitespace-nowrap"
        else:
            classes = "hover:bg-green-700 px-3 py-2 rounded-lg text-sm font-medium transition whitespace-nowrap"
        links_html += f'          <a href="{link_file}" class="{classes}">{name}</a>\n'

    # The HTML block to replace is typically between the logo and the theme toggle
    # Logo snippet: <div class="text-xs text-green-100 tracking-wider">DIGITAL TWIN</div>
    # Theme toggle snippet: <button id="theme-toggle"
    
    # Build a regex to find the links container
    # <div class="... space-x-2 ..."> \n ... </div> \n </div> \n </nav>
    
    pattern = re.compile(
        r'(<div class="[^"]*space-x-2[^"]*">)(.*?)(<div class="h-6 w-px bg-green-400 mx-2)', 
        re.DOTALL
    )
    
    match = pattern.search(content)
    if match:
        new_content = content[:match.start(2)] + '\n' + links_html + '          ' + content[match.start(3):]
        with open(path, 'w', encoding='utf-8') as file:
            file.write(new_content)
        print(f"Updated {f}")
    else:
        print(f"Failed to match pattern in {f}")

