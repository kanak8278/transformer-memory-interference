"""
Quick test: TR AI Platform Workspace → GPT-4.1 Mini
Run: set -a && source .env && set +a && uv run python experiments_cloud/test_openai_tr.py
"""

import os
import sys
import requests

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TR_TOKEN_URL = "https://aiplatform.gcs.int.thomsonreuters.com/v1/openai/token"
MODEL_NAME   = "gpt-4.1-mini"

workspace_id = os.getenv("TR_WORKSPACE_ID")
if not workspace_id:
    print("ERROR: TR_WORKSPACE_ID not set")
    sys.exit(1)

print(f"Workspace ID: {workspace_id}")
print(f"Model:        {MODEL_NAME}")
print(f"Fetching token from {TR_TOKEN_URL} ...")

resp = requests.post(TR_TOKEN_URL, json={"workspace_id": workspace_id, "model_name": MODEL_NAME})
print(f"HTTP status: {resp.status_code}")

if resp.status_code != 200:
    print(f"ERROR: {resp.text}")
    sys.exit(1)

creds = resp.json()
if "openai_key" not in creds:
    print(f"ERROR: unexpected response: {creds}")
    sys.exit(1)

print(f"Token fetched   — valid till: {creds.get('expires_on')}")
print(f"azure_deployment: {creds.get('azure_deployment')}")
print(f"api_version:      {creds.get('openai_api_version')}")

# Test via model interface
from models.openai_model import OpenAIModelInterface
m = OpenAIModelInterface(MODEL_NAME, {'verbose': True})
print(f"Provider: {m.get_model_info()['provider']}")

resp_text = m.generate("Reply with exactly: OK")
print(f"Response: {resp_text.strip()}")
print(f"Tokens in/out: {m.last_input_tokens} / {m.last_output_tokens}")
print("\nSUCCESS: TR OpenAI connection working.")
