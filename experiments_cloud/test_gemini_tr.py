"""
Quick test: TR AI Platform Workspace → Gemini 2.5 Flash
Run: set -a && source .env && set +a && uv run python experiments_cloud/test_gemini_tr.py
"""

import os
import sys
import requests
import json

TR_TOKEN_URL = "https://aiplatform.gcs.int.thomsonreuters.com/v1/gemini/token"
MODEL_NAME   = "gemini-2.5-flash"

workspace_id = os.getenv("TR_WORKSPACE_ID")
if not workspace_id:
    print("ERROR: TR_WORKSPACE_ID not set")
    sys.exit(1)

print(f"Workspace ID: {workspace_id}")
print(f"Model:        {MODEL_NAME}")
print(f"Fetching token from {TR_TOKEN_URL} ...")

# Step 1: fetch token
resp = requests.post(TR_TOKEN_URL, json={"workspace_id": workspace_id, "model_name": MODEL_NAME})
print(f"HTTP status: {resp.status_code}")

if resp.status_code != 200:
    print(f"ERROR: {resp.text}")
    sys.exit(1)

credentials = resp.json()
if "token" not in credentials:
    print(f"ERROR: no token in response: {credentials}")
    sys.exit(1)

print(f"Token fetched — valid till: {credentials.get('expires_on')}")
print(f"project_id: {credentials.get('project_id')}")
print(f"region:     {credentials.get('region')}")

# Step 2: init Vertex AI
try:
    from google.oauth2.credentials import Credentials as OAuth2Credentials
    import vertexai
    from vertexai.generative_models import GenerativeModel
except ImportError:
    print("\nERROR: google-cloud-aiplatform not installed")
    print("Run: uv pip install google-cloud-aiplatform")
    sys.exit(1)

vertexai.init(
    project=credentials["project_id"],
    location=credentials["region"],
    credentials=OAuth2Credentials(credentials["token"]),
)

# Step 3: call model
print("\nCalling model...")
model = GenerativeModel(MODEL_NAME)
response = model.generate_content("Reply with exactly: OK")
print(f"Response: {response.text.strip()}")
print("\nSUCCESS: TR Gemini connection working.")
