import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

print("Attempting to import backend.main...")
try:
    import main
    print("Backend imported successfully.")
except Exception as e:
    print(f"FAILED to import backend: {e}")
    sys.exit(1)

print("Startup check passed.")
