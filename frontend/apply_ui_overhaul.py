import os
import re

FRONTEND_DIR = r"e:\Agri-Twin Digital Twin For Agriculture\frontend"
HTML_FILES = [
    "index.html",
    "live_prices.html", 
    "map.html", 
    "market.html", 
    "profit_calculator.html", 
    "simulation.html", 
    "yield.html"
]

def remove_inline_styles_and_add_link():
    for filename in HTML_FILES:
        filepath = os.path.join(FRONTEND_DIR, filename)
        if not os.path.exists(filepath):
            continue
            
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        
        # We know the old duplicate style block starts with body { font-family: 'Inter', sans-serif; }
        # and goes all the way up to `#page-loader { ... }`.
        # Some files might have additional styles (like index.html has animated sun/cloud).
        # A safer approach: simply inject the <link rel="stylesheet" href="styles.css"> 
        # below the tailwind link if it doesn't exist.
        
        if "styles.css" not in content:
            content = content.replace(
                '<link rel="stylesheet" href="tailwind.min.css">',
                '<link rel="stylesheet" href="tailwind.min.css">\n  <link rel="stylesheet" href="styles.css">'
            )
        
        # We will use regex to remove the specific base dark mode overrides so they don't clash
        # The block usually starts with `/* Base typography */` (or just `body {`) and ends around the `html.dark #page-loader` rule.
        
        # Regex to match the common redundant block which is inside <style>
        # We have to be careful not to delete leaflet styles or animation css.
        pattern = re.compile(
            r'/\*\s*Base typography\s*\*/.*?html\.dark\s+#page-loader\s*\{\s*background-color:\s*#111827;\s*\}', 
            re.DOTALL
        )
        
        matches = pattern.findall(content)
        if matches:
            content = content.replace(matches[0], "/* Styles migrated to styles.css */\n")
            print(f"Migrated styles for {filename}")
        else:
            # Fallback if the pattern doesn't exactly match (it might not have page-loader in some pages)
            pattern2 = re.compile(
                r'/\*\s*Base typography\s*\*/.*?html\.dark\s+\.text-gray-400\s*\{\s*color:\s*#9ca3af\s*!important;\s*\}', 
                re.DOTALL
            )
            matches2 = pattern2.findall(content)
            if matches2:
                content = content.replace(matches2[0], "/* Styles migrated to styles.css */\n")
                print(f"Migrated styles (pattern 2) for {filename}")
            else:
                print(f"Could not find style block to replace in {filename}, injecting link manually.")

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

if __name__ == "__main__":
    remove_inline_styles_and_add_link()
