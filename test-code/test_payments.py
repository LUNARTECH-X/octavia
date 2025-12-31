import requests
import json

BASE_URL = "http://localhost:8000"

# 1. Get demo login
print("1. Getting demo user...")
login_response = requests.post(f"{BASE_URL}/api/auth/demo-login", 
                              json={"email": "demo@octavia.com", "password": "demo123"})
token = login_response.json()["token"]
print(f"Token: {token[:20]}...")

# 2. Get packages
print("\n2. Getting credit packages...")
packages_response = requests.get(f"{BASE_URL}/api/payments/packages")
print(f"Packages available: {len(packages_response.json()['packages'])}")

# 3. Get current credits
print("\n3. Getting current credits...")
credits_response = requests.get(f"{BASE_URL}/api/user/credits",
                               headers={"Authorization": f"Bearer {token}"})
current_credits = credits_response.json()["credits"]
print(f"Current credits: {current_credits}")

# 4. "Purchase" starter package
print("\n4. Purchasing starter credits package...")
purchase_response = requests.post(f"{BASE_URL}/api/payments/create-session",
                                 headers={"Authorization": f"Bearer {token}"},
                                 json={"package_id": "starter_credits"})
print(f"Purchase result: {purchase_response.json()}")

# 5. Check new credits
print("\n5. Verifying new credit balance...")
new_credits_response = requests.get(f"{BASE_URL}/api/user/credits",
                                   headers={"Authorization": f"Bearer {token}"})
new_credits = new_credits_response.json()["credits"]
print(f"New credits: {new_credits} (increase: {new_credits - current_credits})")

# 6. Check transactions
print("\n6. Checking transaction history...")
transactions_response = requests.get(f"{BASE_URL}/api/payments/transactions",
                                    headers={"Authorization": f"Bearer {token}"})
print(f"Transactions: {len(transactions_response.json()['transactions'])} found")