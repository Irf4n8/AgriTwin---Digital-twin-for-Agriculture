import os

files = ['index.html', 'live_prices.html', 'map.html', 'market.html', 'simulation.html', 'yield.html']
target = '<div class="h-6 w-px bg-green-400 mx-2 hidden md:block"></div>'
replacement = '<a href="profit_calculator.html"\n            class="hover:bg-green-700 px-3 py-2 rounded-lg text-sm font-medium transition whitespace-nowrap">Profit Calculator</a>\n          ' + target

for f in files:
    path = os.path.join(r'e:\Agri-Twin Digital Twin For Agriculture\frontend', f)
    with open(path, 'r', encoding='utf-8') as file:
        content = file.read()
    if 'profit_calculator.html' not in content:
        if target in content:
            new_content = content.replace(target, replacement)
            with open(path, 'w', encoding='utf-8') as file:
                file.write(new_content)
            print(f'Updated {f}')
        else:
            print(f'Target not found in {f}')
