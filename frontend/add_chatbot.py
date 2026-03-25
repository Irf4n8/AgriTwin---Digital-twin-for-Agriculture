import os
import re

index_path = 'index.html'

with open(index_path, 'r', encoding='utf-8') as f:
    index_content = f.read()

# 1. Extract CSS
css_match = re.search(r'(#chatbot-fab \{[\s\S]*?)(?=</style>)', index_content)
chatbot_css = css_match.group(1) if css_match else ""

# 2. Extract HTML
html_match = re.search(r'(<!-- RAG Chatbot UI -->[\s\S]*?</form>\s*</div>)', index_content)
chatbot_html = html_match.group(1) if html_match else ""

# 3. Extract JS
js_match = re.search(r'(    // --- Chatbot Logic ---[\s\S]*?)(?=  </script>)', index_content)
chatbot_js = js_match.group(1) if js_match else ""

if not chatbot_css or not chatbot_html or not chatbot_js:
    print("Could not find blocks in index.html! Check regex.")
    exit(1)

html_files = [f for f in os.listdir('.') if f.endswith('.html') and f != 'index.html']

for target_file in html_files:
    with open(target_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Skip if already injected
    if 'id="chatbot-fab"' in content:
        print(f"Skipping {target_file}, chatbot already exists.")
        continue

    # Insert CSS just before </style>
    if '</style>' in content:
        content = content.replace('</style>', '\n' + chatbot_css + '\n</style>', 1)
    else:
        content = content.replace('</head>', '\n<style>\n' + chatbot_css + '\n</style>\n</head>', 1)

    # Insert HTML just before </body> 
    content = content.replace('</body>', '\n' + chatbot_html + '\n</body>', 1)

    # Prepare JS
    js_to_insert = chatbot_js
    if "API_BASE" not in content:
        js_to_insert = '\nconst API_BASE = "http://127.0.0.1:8000";\n' + js_to_insert

    # Insert JS just before the last </script>
    last_script_idx = content.rfind('</script>')
    if last_script_idx != -1:
        content = content[:last_script_idx] + '\n' + js_to_insert + '\n' + content[last_script_idx:]
    else:
        # If no script at all (rare)
        content = content.replace('</body>', '\n<script>\n' + js_to_insert + '\n</script>\n</body>', 1)

    with open(target_file, 'w', encoding='utf-8') as f:
        f.write(content)
        
    print(f"Injected into {target_file}")
