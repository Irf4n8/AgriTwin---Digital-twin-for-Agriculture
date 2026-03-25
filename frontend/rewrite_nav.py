import os
import re

files = ['index.html', 'market.html', 'live_prices.html', 'map.html', 'simulation.html', 'yield.html', 'profit_calculator.html']
base_dir = r'e:\Agri-Twin Digital Twin For Agriculture\frontend'

nav_template = """  <!-- Navigation -->
  <nav class="glass sticky top-0 z-50 shadow-lg text-white">
    <div class="max-w-7xl mx-auto px-4">
      <div class="flex items-center justify-between h-16">
        <!-- Logo -->
        <div class="flex items-center gap-3 cursor-pointer hover-lift" onclick="window.location.href='index.html'">
          <span class="text-3xl drop-shadow-md">🌱</span>
          <div>
            <div class="font-bold text-xl leading-none outfit tracking-tight">AgriTwin</div>
            <div class="text-[10px] text-green-200 tracking-widest uppercase font-semibold mt-0.5">Digital Twin</div>
          </div>
        </div>

        <!-- Links -->
        <div class="flex items-center space-x-1 overflow-x-auto no-scrollbar pb-2 md:pb-0 w-full md:w-auto">
{LINKS}          <div class="h-6 w-px bg-white/20 mx-3 hidden md:block"></div>

          <!-- Telegram Bot Quick Button -->
          <a href="https://t.me/agritwin_bot" target="_blank" class="hidden md:flex items-center gap-2 mr-3 bg-white/10 hover:bg-white/20 border border-white/10 px-3 py-1.5 rounded-full text-xs font-medium transition hover-lift">
             <span>💬</span> Bot
          </a>

          <!-- Theme Toggle -->
          <button id="theme-toggle" class="p-2 rounded-full hover:bg-white/10 transition flex-shrink-0 border border-transparent hover:border-white/10"
            title="Toggle Theme">
            <svg id="theme-toggle-light-icon" class="w-5 h-5 hidden" fill="none" stroke="currentColor"
              viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z">
              </path>
            </svg>
            <svg id="theme-toggle-dark-icon" class="w-5 h-5 hidden" fill="none" stroke="currentColor"
              viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z"></path>
            </svg>
          </button>
        </div>
      </div>
    </div>
  </nav>"""

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
            classes = "bg-white/20 px-4 py-2 rounded-lg text-sm font-bold shadow-sm whitespace-nowrap text-white border border-white/10"
        else:
            classes = "hover:bg-white/10 px-3 py-2 rounded-lg text-sm font-medium transition whitespace-nowrap text-green-50"
        links_html += f'          <a href="{link_file}" class="{classes}">{name}</a>\n'

    full_nav = nav_template.replace('{LINKS}', links_html)

    # Replace the entire nav block in the file
    pattern = re.compile(r'<nav class="glass.*?<\/nav>', re.DOTALL)
    
    if pattern.search(content):
        new_content = pattern.sub(full_nav, content)
        with open(path, 'w', encoding='utf-8') as file:
            file.write(new_content)
        print(f"Updated entire nav in {f}")
    else:
        print(f"Failed to match <nav> in {f}")
