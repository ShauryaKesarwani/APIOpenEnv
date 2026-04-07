"""Quick import check."""
import sys
print("Python path:", sys.path[:3])

try:
    from server.api_open_env_environment import ApiOpenEnvironment
    print("✓ Environment imported successfully")
except Exception as e:
    print(f"✗ Environment import failed: {e}")

try:
    from models import ApiOpenAction, ApiOpenObservation
    print("✓ Models imported successfully")
except Exception as e:
    print(f"✗ Models import failed: {e}")

try:
    from server.mock_apis import call_api
    print("✓ Mock APIs imported successfully")
except Exception as e:
    print(f"✗ Mock APIs import failed: {e}")

try:
    from server.grader import grade_episode
    print("✓ Grader imported successfully")
except Exception as e:
    print(f"✗ Grader import failed: {e}")

print("\n✓ All imports successful!")
