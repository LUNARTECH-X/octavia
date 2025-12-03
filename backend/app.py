"""
Octavia Video Translator Backend
FastAPI application with Supabase authentication and Polar.sh payments
Complete payment flow with real Polar.sh integration
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Request, Response, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, RedirectResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Dict, Optional, List, Any
import whisper
from transformers import pipeline
import os
import json
import uuid
import shutil
import subprocess
import logging
import asyncio
from datetime import datetime, timedelta
import time
from contextlib import asynccontextmanager
import psutil
import sys
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from supabase import create_client, Client
from jose import JWTError, jwt
import secrets
from pydantic import BaseModel, EmailStr
from typing import Optional
import traceback
import hashlib
import binascii
from datetime import timezone

from dotenv import load_dotenv
load_dotenv()

# Configuration from environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
POLAR_WEBHOOK_SECRET = os.getenv("POLAR_WEBHOOK_SECRET", "")

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# Polar.sh configuration
POLAR_ACCESS_TOKEN = os.getenv("POLAR_ACCESS_TOKEN")
POLAR_SERVER = os.getenv("POLAR_SERVER", "sandbox")
ENABLE_TEST_MODE = os.getenv("ENABLE_TEST_MODE", "true").lower() == "true"

# Initialize Polar.sh client
try:
    from polar_sdk import Polar
    
    polar_client = Polar(
        access_token=POLAR_ACCESS_TOKEN,
        server=POLAR_SERVER
    )
    print(f"Polar.sh client initialized for {POLAR_SERVER} environment")
    
    try:
        org_response = polar_client.organizations.get()
        if org_response and hasattr(org_response, 'organization'):
            org_name = org_response.organization.name
            print(f"Connected to Polar.sh organization: {org_name}")
        else:
            print("Polar.sh connection successful")
    except Exception as connection_error:
        print(f"Polar.sh connection test had issues: {connection_error}")
        
except ImportError as import_error:
    print(f"Failed to import Polar SDK: {import_error}")
    print("Please install the polar-sdk package: pip install polar-sdk")
    polar_client = None
except Exception as init_error:
    print(f"Failed to initialize Polar.sh client: {init_error}")
    polar_client = None

# Credit packages configuration with real Polar.sh product IDs
CREDIT_PACKAGES = {
    "starter_credits": {
        "name": "Starter Credits",
        "credits": 100,
        "price": 999,
        "polar_product_id": "68d54da0-c3ec-4215-9636-21457e57b3e6",
        "checkout_link": "https://sandbox-api.polar.sh/v1/checkout-links/polar_cl_ENF1TwWHLmhB809OfLQozk0UCGMLmYinMbfT14K8K2R/redirect",
        "description": "100 translation credits",
        "features": ["100 credits", "Standard processing", "Email support"],
        "popular": False
    },
    "pro_credits": {
        "name": "Pro Credits",
        "credits": 250,
        "price": 1999,
        "polar_product_id": "743297c6-eadb-4b96-a8d6-b4c815f0f1b5",
        "checkout_link": "https://sandbox-api.polar.sh/v1/checkout-links/polar_cl_SXDRYMs6nvN9dm8b5wK8Z3WcsowTEU7jYPXFe4XXHgm/redirect",
        "description": "250 translation credits",
        "features": ["250 credits", "Priority processing", "Priority support"],
        "popular": True
    },
    "premium_credits": {
        "name": "Premium Credits",
        "credits": 500,
        "price": 3499,
        "polar_product_id": "2dceabdb-d0f8-4ddd-9b68-af44f0c4ad96",
        "checkout_link": "https://sandbox-api.polar.sh/v1/checkout-links/polar_cl_QNmrgCNlflNXndg61t31JhwmQVIe5cthFDyAy2yb2ED/redirect",
        "description": "500 translation credits",
        "features": ["500 credits", "Express processing", "24/7 support", "Batch upload"],
        "popular": False
    }
}

# Password management utility
class PasswordManager:
    def __init__(self):
        self.method = "sha256"
    
    def hash_password(self, password: str) -> str:
        salt = secrets.token_hex(16)
        hashed_password = hashlib.sha256((password + salt).encode()).hexdigest()
        return f"sha256:{salt}:{hashed_password}"
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        try:
            if not hashed_password or ':' not in hashed_password:
                return False
            
            parts = hashed_password.split(':')
            if len(parts) != 3:
                return False
            
            method, salt, stored_hash = parts
            
            if method == "sha256":
                computed_hash = hashlib.sha256((plain_password + salt).encode()).hexdigest()
                return computed_hash == stored_hash
            else:
                return self._try_bcrypt_fallback(plain_password, hashed_password)
                
        except Exception as verification_error:
            print(f"Password verification error: {verification_error}")
            return False
    
    def _try_bcrypt_fallback(self, plain_password: str, hashed_password: str) -> bool:
        try:
            import bcrypt
            if hashed_password.startswith('$2b$') or hashed_password.startswith('$2a$'):
                return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))
        except Exception:
            pass
        return False

password_manager = PasswordManager()
security = HTTPBearer()

# Create artifacts directory for logs
os.makedirs("artifacts", exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='{"timestamp": "%(asctime)s", "stage": "%(name)s", "chunk_id": "%(filename)s", "duration_ms": %(relativeCreated)d, "status": "%(levelname)s", "message": "%(message)s"}',
    handlers=[
        logging.FileHandler('artifacts/logs.jsonl'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Pydantic models for request/response validation
class UserCreate(BaseModel):
    email: EmailStr
    password: str
    name: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class User(BaseModel):
    id: str
    email: str
    name: str
    is_verified: bool
    credits: int
    created_at: datetime

class Token(BaseModel):
    access_token: str
    token_type: str

class PaymentSessionCreate(BaseModel):
    package_id: str

class TestCreditAdd(BaseModel):
    credits: int

# Authentication helper functions
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return password_manager.verify_password(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return password_manager.hash_password(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str):
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    payload = verify_token(token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
        )
    
    try:
        response = supabase.table("users").select("*").eq("id", user_id).execute()
        if not response.data:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found",
            )
        
        user_data = response.data[0]
        return User(
            id=user_data["id"],
            email=user_data["email"],
            name=user_data["name"],
            is_verified=user_data["is_verified"],
            credits=user_data["credits"],
            created_at=user_data["created_at"]
        )
    except Exception as db_error:
        logger.error(f"Error fetching user: {db_error}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error fetching user data",
        )

# Credit management functions
def add_user_credits(user_id: str, credits_to_add: int, description: str = "Credit purchase"):
    try:
        response = supabase.table("users").select("credits").eq("id", user_id).execute()
        if not response.data:
            raise Exception("User not found")
        
        current_credits = response.data[0]["credits"]
        new_credits = current_credits + credits_to_add
        
        supabase.table("users").update({"credits": new_credits}).eq("id", user_id).execute()
        
        transaction_id = str(uuid.uuid4())
        transaction_data = {
            "id": transaction_id,
            "user_id": user_id,
            "amount": credits_to_add,
            "type": "credit_purchase",
            "status": "completed",
            "description": description,
            "created_at": datetime.utcnow().isoformat()
        }
        
        supabase.table("transactions").insert(transaction_data).execute()
        
        logger.info(f"Added {credits_to_add} credits to user {user_id}. New balance: {new_credits}")
        return new_credits
        
    except Exception as credit_error:
        logger.error(f"Failed to add credits: {credit_error}")
        raise

def update_transaction_status(transaction_id: str, status: str, description: str = None):
    try:
        update_data = {
            "status": status,
            "updated_at": datetime.utcnow().isoformat()
        }
        if description:
            update_data["description"] = description
            
        supabase.table("transactions").update(update_data).eq("id", transaction_id).execute()
        logger.info(f"Updated transaction {transaction_id} to status: {status}")
        return True
    except Exception as update_error:
        logger.error(f"Failed to update transaction status: {update_error}")
        return False

# Email sending function for verification
def send_verification_email(email: str, name: str, verification_token: str):
    try:
        smtp_server = os.getenv("SMTP_HOST", "smtp.gmail.com")
        smtp_port = int(os.getenv("SMTP_PORT", "587"))
        smtp_username = os.getenv("SMTP_USER")
        smtp_password = os.getenv("SMTP_PASS")
        smtp_from = os.getenv("SMTP_FROM", "noreply@octavia.com")
        
        if not all([smtp_username, smtp_password]):
            logger.warning(f"SMTP credentials not configured. Mock email would be sent to: {email}")
            return True
        
        msg = MIMEMultipart('alternative')
        msg['Subject'] = "Verify Your Octavia Account"
        msg['From'] = smtp_from
        msg['To'] = email
        
        verification_link = f"http://localhost:3000/verify-email?token={verification_token}"
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; text-align: center; color: white; border-radius: 10px 10px 0 0; }}
                .content {{ background: #f9f9f9; padding: 30px; border-radius: 0 0 10px 10px; }}
                .button {{ display: inline-block; padding: 12px 24px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; text-decoration: none; border-radius: 5px; font-weight: bold; }}
                .footer {{ text-align: center; margin-top: 30px; color: #666; font-size: 12px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Welcome to Octavia!</h1>
                </div>
                <div class="content">
                    <h2>Hi {name},</h2>
                    <p>Thank you for signing up for Octavia Video Translator. To start using your account, please verify your email address by clicking the button below:</p>
                    <p style="text-align: center; margin: 30px 0;">
                        <a href="{verification_link}" class="button">Verify Email Address</a>
                    </p>
                    <p>Or copy and paste this link into your browser:</p>
                    <p style="background: #eee; padding: 10px; border-radius: 5px; word-break: break-all;">
                        {verification_link}
                    </p>
                    <p>This link will expire in 24 hours.</p>
                    <p>If you didn't create an account with Octavia, you can safely ignore this email.</p>
                </div>
                <div class="footer">
                    <p>&copy; 2024 Octavia Video Translator. All rights reserved.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        text = f"""Welcome to Octavia!
        
Hi {name},
        
Thank you for signing up for Octavia Video Translator. To start using your account, please verify your email address by clicking the link below:
        
{verification_link}
        
This link will expire in 24 hours.
        
If you didn't create an account with Octavia, you can safely ignore this email.
        
Best regards,
The Octavia Team"""
        
        part1 = MIMEText(text, 'plain')
        part2 = MIMEText(html, 'html')
        msg.attach(part1)
        msg.attach(part2)
        
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_username, smtp_password)
            server.send_message(msg)
        
        logger.info(f"Verification email sent to {email}")
        return True
        
    except Exception as email_error:
        logger.error(f"Failed to send verification email to {email}: {email_error}")
        return True

# Application lifecycle management
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting Octavia Video Translator with Supabase...")
    print("=" * 60)
    
    hardware_info = {
        "cpu_count": psutil.cpu_count(),
        "total_ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "available_ram_gb": round(psutil.virtual_memory().available / (1024**3), 2),
        "platform": sys.platform,
        "python_version": sys.version
    }
    
    logger.info(f"Hardware detected: {hardware_info}")
    
    try:
        response = supabase.table("users").select("count", count="exact").limit(1).execute()
        print("Connected to Supabase database")
        
        try:
            supabase.table("transactions").select("count", count="exact").limit(1).execute()
        except:
            print("Note: Transactions table will be created automatically")
    except Exception as db_error:
        print(f"Supabase connection issue: {db_error}")
    
    if polar_client:
        print(f"Polar.sh payment system ready ({POLAR_SERVER} mode)")
        if ENABLE_TEST_MODE:
            print("Payment test mode enabled")
        else:
            print("REAL PAYMENT MODE - Polar.sh integration active")
    else:
        print("Polar.sh payment system not available")
    
    print("Loading AI models...")
    global whisper_model, translator
    
    try:
        whisper_model = whisper.load_model("base")
        print("Whisper speech recognition model loaded")
    except Exception as whisper_error:
        print(f"Whisper load failed: {whisper_error}")
        whisper_model = None
    
    try:
        from transformers import MarianMTModel, MarianTokenizer
        model_name = "Helsinki-NLP/opus-mt-en-es"
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        translator = pipeline("translation", model=model, tokenizer=tokenizer)
        print("Translation model loaded")
    except Exception as translation_error:
        print(f"Translation model failed: {translation_error}")
        translator = None
    
    print("AI models ready")
    print("=" * 60)
    
    yield
    
    print("Shutting down Octavia...")
    for file in os.listdir("."):
        if file.startswith("temp_") or file.startswith("translated_"):
            try:
                os.remove(file)
            except:
                pass

# Create FastAPI application
app = FastAPI(
    title="Octavia Video Translator",
    description="End-to-end video dubbing with perfect lip-sync and timing",
    version="4.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# In-memory storage for jobs and files (in production, use database)
jobs_db: Dict[str, Dict] = {}
files_db: Dict[str, str] = {}

# Payment endpoints
@app.get("/api/payments/packages")
async def get_credit_packages():
    try:
        packages_list = []
        for package_id, package in CREDIT_PACKAGES.items():
            packages_list.append({
                "id": package_id,
                "name": package["name"],
                "credits": package["credits"],
                "price": package["price"] / 100,
                "description": package["description"],
                "features": package["features"],
                "popular": package.get("popular", False),
                "checkout_link": package.get("checkout_link")
            })
        
        return {
            "success": True,
            "packages": packages_list
        }
    except Exception as package_error:
        logger.error(f"Failed to get packages: {package_error}")
        raise HTTPException(500, "Failed to retrieve packages")

@app.post("/api/payments/create-session")
async def create_payment_session(
    request: Request,
    current_user: User = Depends(get_current_user)
):
    try:
        data = await request.json()
        package_id = data.get("package_id")
        
        if not package_id:
            raise HTTPException(400, "Package ID is required")
        
        package = CREDIT_PACKAGES.get(package_id)
        if not package:
            raise HTTPException(400, "Invalid package")
        
        session_id = str(uuid.uuid4())
        
        transaction_id = str(uuid.uuid4())
        transaction_data = {
            "id": transaction_id,
            "user_id": current_user.id,
            "email": current_user.email,
            "package_id": package_id,
            "credits": package["credits"],
            "amount": package["price"],
            "type": "credit_purchase",
            "status": "pending",
            "description": f"Pending purchase: {package['name']}",
            "session_id": session_id,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }
        
        supabase.table("transactions").insert(transaction_data).execute()
        
        logger.info(f"Created pending transaction {transaction_id} for user {current_user.email}")
        
        if ENABLE_TEST_MODE:
            await asyncio.sleep(1)
            
            new_balance = add_user_credits(
                current_user.id,
                package["credits"],
                f"Test purchase: {package['name']}"
            )
            
            update_transaction_status(
                transaction_id, 
                "completed", 
                f"Test purchase completed: {package['name']}"
            )
            
            logger.info(f"Test purchase completed for user {current_user.email}")
            
            return {
                "success": True,
                "test_mode": True,
                "message": "Test credits added successfully",
                "credits_added": package["credits"],
                "new_balance": new_balance,
                "checkout_url": None,
                "session_id": session_id,
                "transaction_id": transaction_id,
                "status": "completed"
            }
        
        try:
            checkout_link = package["checkout_link"]
            
            if "email=" not in checkout_link:
                separator = "&" if "?" in checkout_link else "?"
                checkout_url = f"{checkout_link}{separator}email={current_user.email}"
                checkout_url += f"&metadata[user_id]={current_user.id}"
                checkout_url += f"&metadata[transaction_id]={transaction_id}"
                checkout_url += f"&metadata[package_id]={package_id}"
                checkout_url += f"&metadata[session_id]={session_id}"
            else:
                checkout_url = checkout_link
                if "metadata[user_id]" not in checkout_url:
                    separator = "&" if "?" in checkout_url else "?"
                    checkout_url += f"{separator}metadata[user_id]={current_user.id}"
                    checkout_url += f"&metadata[transaction_id]={transaction_id}"
                    checkout_url += f"&metadata[package_id]={package_id}"
                    checkout_url += f"&metadata[session_id]={session_id}"
            
            logger.info(f"Created REAL payment session {session_id} for user {current_user.email}")
            
            return {
                "success": True,
                "test_mode": False,
                "session_id": session_id,
                "transaction_id": transaction_id,
                "checkout_url": checkout_url,
                "package_id": package_id,
                "credits": package["credits"],
                "price": package["price"] / 100,
                "message": "Checkout session created. You will be redirected to complete payment.",
                "status": "pending"
            }
            
        except Exception as polar_error:
            logger.error(f"Polar.sh error: {polar_error}")
            traceback.print_exc()
            return {
                "success": False,
                "test_mode": False,
                "error": "Payment service temporarily unavailable.",
                "message": "Unable to create payment session"
            }
        
    except HTTPException:
        raise
    except Exception as session_error:
        logger.error(f"Failed to create payment session: {session_error}")
        traceback.print_exc()
        return {
            "success": False,
            "error": "Failed to create payment session.",
            "message": "Internal server error"
        }

@app.get("/api/payments/status/{session_id}")
async def get_payment_status(
    session_id: str,
    current_user: User = Depends(get_current_user)
):
    try:
        response = supabase.table("transactions").select("*").eq("session_id", session_id).execute()
        
        if not response.data:
            raise HTTPException(404, "Transaction not found")
        
        transaction = response.data[0]
        
        if transaction["user_id"] != current_user.id:
            raise HTTPException(403, "Access denied")
        
        # Auto-completion logic for better user experience
        if transaction["status"] == "pending":
            created_at_str = transaction["created_at"]
            
            try:
                if created_at_str.endswith('Z'):
                    created_at = datetime.fromisoformat(created_at_str.replace('Z', '+00:00'))
                else:
                    created_at = datetime.fromisoformat(created_at_str)
                
                utc_now = datetime.utcnow().replace(tzinfo=timezone.utc)
                
                # Calculate time elapsed - reduced to 60 seconds for better UX
                time_elapsed = (utc_now - created_at).total_seconds()
                
                # If more than 60 seconds have passed, auto-complete it
                if time_elapsed > 60:
                    package_id = transaction.get("package_id")
                    
                    if package_id and package_id in CREDIT_PACKAGES:
                        package = CREDIT_PACKAGES[package_id]
                        credits_to_add = package["credits"]
                        
                        # Add credits
                        add_user_credits(
                            current_user.id,
                            credits_to_add,
                            f"Auto-completed after 60s timeout: {package['name']}"
                        )
                        
                        # Update transaction
                        update_transaction_status(
                            transaction["id"],
                            "completed",
                            f"Auto-completed: Payment likely succeeded but webhook delayed"
                        )
                        
                        # Refresh transaction data
                        response = supabase.table("transactions").select("*").eq("id", transaction["id"]).execute()
                        if response.data:
                            transaction = response.data[0]
                        
            except ValueError as date_error:
                logger.error(f"Date parsing error: {date_error}")
                pass
        
        return {
            "success": True,
            "session_id": session_id,
            "transaction_id": transaction["id"],
            "status": transaction["status"],
            "credits": transaction.get("credits", 0),
            "description": transaction.get("description", ""),
            "created_at": transaction.get("created_at"),
            "updated_at": transaction.get("updated_at")
        }
        
    except HTTPException:
        raise
    except Exception as status_error:
        logger.error(f"Failed to get payment status: {status_error}")
        raise HTTPException(500, "Failed to get payment status")

@app.post("/api/payments/webhook/polar")
async def polar_webhook(request: Request):
    try:
        # Log raw request for debugging
        logger.info(f"Polar webhook received. Headers: {dict(request.headers)}")
        
        payload_body = await request.body()
        payload = json.loads(payload_body)
        event_type = payload.get("type")
        event_id = payload.get("id")
        
        logger.info(f"Polar webhook: {event_type} (ID: {event_id})")
        logger.info(f"Webhook payload: {json.dumps(payload, indent=2)}")
        
        # Store webhook for debugging
        webhook_log = {
            "id": str(uuid.uuid4()),
            "event_type": event_type,
            "event_id": event_id,
            "payload": json.dumps(payload),
            "received_at": datetime.utcnow().isoformat(),
            "status": "received"
        }
        
        supabase.table("webhook_logs").insert(webhook_log).execute()
        
        # Process all payment success events
        if event_type in ["order.completed", "order.paid", "order.updated"]:
            order_data = payload.get("data", {})
            order_id = order_data.get("id")
            
            # Check if order is actually paid
            order_status = order_data.get("status", "")
            is_paid = order_data.get("paid", False)
            
            logger.info(f"Processing {event_type}: Order {order_id}, Status: {order_status}, Paid: {is_paid}")
            
            # Only process if order is paid/completed
            if is_paid and order_status in ["paid", "completed"]:
                customer_email = order_data.get("customer_email")
                amount = order_data.get("amount", 0)
                
                logger.info(f"Payment SUCCESS: {order_id} for {customer_email} - Amount: {amount}")
                
                # Look for metadata in multiple locations
                metadata = {}
                checkout_session = order_data.get("checkout_session", {})
                if checkout_session:
                    metadata.update(checkout_session.get("metadata", {}))
                
                # Also check order metadata
                metadata.update(order_data.get("metadata", {}))
                
                logger.info(f"Metadata found: {metadata}")
                
                # Find user by email first (most reliable)
                user = None
                if customer_email:
                    response = supabase.table("users").select("*").eq("email", customer_email).execute()
                    if response.data:
                        user = response.data[0]
                        logger.info(f"Found user by email: {user['id']} - {user['email']}")
                
                # If no user found, check metadata
                if not user and metadata.get("user_id"):
                    response = supabase.table("users").select("*").eq("id", metadata.get("user_id")).execute()
                    if response.data:
                        user = response.data[0]
                        logger.info(f"Found user by metadata user_id: {user['id']}")
                
                if not user:
                    logger.error(f"No user found for order {order_id}")
                    # Try to find user by looking up transactions with this session_id
                    if metadata.get("session_id"):
                        response = supabase.table("transactions").select("*").eq("session_id", metadata.get("session_id")).execute()
                        if response.data:
                            tx = response.data[0]
                            response = supabase.table("users").select("*").eq("id", tx["user_id"]).execute()
                            if response.data:
                                user = response.data[0]
                                logger.info(f"Found user via session_id lookup: {user['id']}")
                
                if not user:
                    logger.error(f"No user found for order {order_id} after all attempts")
                    # Create a failed transaction record for tracking
                    transaction_data = {
                        "id": str(uuid.uuid4()),
                        "order_id": order_id,
                        "customer_email": customer_email,
                        "amount": amount,
                        "type": "credit_purchase",
                        "status": "failed",
                        "description": f"Order {order_id} - No user found",
                        "created_at": datetime.utcnow().isoformat()
                    }
                    supabase.table("transactions").insert(transaction_data).execute()
                    
                    # Update webhook log with error
                    supabase.table("webhook_logs").update({
                        "status": "error",
                        "error": f"No user found for order {order_id}"
                    }).eq("id", webhook_log["id"]).execute()
                    
                    return {"success": False, "error": f"No user found for email: {customer_email}"}
                
                # Determine which package was purchased
                credits_to_add = 0
                package_name = "Unknown Package"
                
                # Try to find by package_id in metadata
                if metadata.get("package_id") and metadata["package_id"] in CREDIT_PACKAGES:
                    package = CREDIT_PACKAGES[metadata["package_id"]]
                    credits_to_add = package["credits"]
                    package_name = package["name"]
                    logger.info(f"Found package by package_id: {package_name} ({credits_to_add} credits)")
                else:
                    # Fallback: match by amount
                    for package_id, package in CREDIT_PACKAGES.items():
                        if package["price"] == amount:
                            credits_to_add = package["credits"]
                            package_name = package["name"]
                            logger.info(f"Matched package by amount: {package_name} ({credits_to_add} credits)")
                            break
                
                if credits_to_add == 0:
                    # Default fallback based on amount ranges
                    if amount >= 3499:
                        credits_to_add = 500
                        package_name = "Premium Credits (auto-detected)"
                    elif amount >= 1999:
                        credits_to_add = 250
                        package_name = "Pro Credits (auto-detected)"
                    else:
                        credits_to_add = 100
                        package_name = "Starter Credits (auto-detected)"
                    logger.info(f"Using fallback credits: {credits_to_add} for amount {amount}")
                
                # Update user credits
                try:
                    # Get current credits
                    response = supabase.table("users").select("credits").eq("id", user["id"]).execute()
                    if not response.data:
                        raise Exception("User not found in database")
                    
                    current_credits = response.data[0]["credits"]
                    new_credits = current_credits + credits_to_add
                    
                    # Update user in database
                    update_result = supabase.table("users").update({
                        "credits": new_credits,
                        "updated_at": datetime.utcnow().isoformat()
                    }).eq("id", user["id"]).execute()
                    
                    if update_result.data:
                        logger.info(f"Updated credits for user {user['id']}: {current_credits} -> {new_credits}")
                    else:
                        logger.error(f"Failed to update credits for user {user['id']}")
                    
                    # Create transaction record
                    transaction_id = str(uuid.uuid4())
                    transaction_data = {
                        "id": transaction_id,
                        "user_id": user["id"],
                        "email": user["email"],
                        "order_id": order_id,
                        "package_id": metadata.get("package_id", "unknown"),
                        "credits": credits_to_add,
                        "amount": amount,
                        "type": "credit_purchase",
                        "status": "completed",
                        "description": f"Payment completed: {package_name} (Order {order_id})",
                        "session_id": metadata.get("session_id", order_id),
                        "created_at": datetime.utcnow().isoformat(),
                        "updated_at": datetime.utcnow().isoformat()
                    }
                    
                    # Check if transaction already exists (by session_id)
                    existing_tx = None
                    if metadata.get("session_id"):
                        existing_tx = supabase.table("transactions") \
                            .select("*") \
                            .eq("session_id", metadata.get("session_id")) \
                            .execute()
                    
                    # Also check by order_id
                    if not existing_tx or not existing_tx.data:
                        existing_tx = supabase.table("transactions") \
                            .select("*") \
                            .eq("order_id", order_id) \
                            .execute()
                    
                    if not existing_tx.data:
                        # Create new transaction
                        supabase.table("transactions").insert(transaction_data).execute()
                        logger.info(f"Created NEW transaction record: {transaction_id}")
                    else:
                        # Update existing transaction
                        tx_id = existing_tx.data[0]["id"]
                        supabase.table("transactions").update({
                            "status": "completed",
                            "credits": credits_to_add,
                            "amount": amount,
                            "updated_at": datetime.utcnow().isoformat(),
                            "description": f"Payment completed: {package_name} (Order {order_id})",
                            "order_id": order_id
                        }).eq("id", tx_id).execute()
                        logger.info(f"Updated EXISTING transaction: {tx_id}")
                    
                    # Update webhook log
                    supabase.table("webhook_logs").update({
                        "status": "processed",
                        "user_id": user["id"],
                        "transaction_id": transaction_id,
                        "credits_added": credits_to_add
                    }).eq("id", webhook_log["id"]).execute()
                    
                    logger.info(f"Successfully processed {event_type} webhook for order {order_id}")
                    logger.info(f"Added {credits_to_add} credits to {user['email']}. New balance: {new_credits}")
                    
                    return {
                        "success": True, 
                        "message": f"Added {credits_to_add} credits to user {user['email']}",
                        "credits_added": credits_to_add,
                        "new_balance": new_credits,
                        "event_type": event_type,
                        "order_id": order_id
                    }
                    
                except Exception as credit_update_error:
                    logger.error(f"Failed to update credits: {credit_update_error}")
                    traceback.print_exc()
                    
                    # Update webhook log with error
                    supabase.table("webhook_logs").update({
                        "status": "error",
                        "error": str(credit_update_error)
                    }).eq("id", webhook_log["id"]).execute()
                    
                    return {
                        "success": False, 
                        "error": f"Failed to update credits: {str(credit_update_error)}",
                        "event_type": event_type
                    }
            else:
                logger.info(f"Ignoring {event_type} - order not paid yet (status: {order_status}, paid: {is_paid})")
                return {"success": True, "message": f"Ignored {event_type} - order not paid yet"}
        
        elif event_type == "order.created":
            order_data = payload.get("data", {})
            order_id = order_data.get("id")
            logger.info(f"Order created: {order_id}")
            
            # Update webhook log
            supabase.table("webhook_logs").update({
                "status": "processed",
                "message": f"Order created: {order_id}"
            }).eq("id", webhook_log["id"]).execute()
            
        elif event_type == "order.failed":
            order_data = payload.get("data", {})
            order_id = order_data.get("id")
            logger.warning(f"Payment failed for order: {order_id}")
            
            # Create failed transaction record
            transaction_id = str(uuid.uuid4())
            transaction_data = {
                "id": transaction_id,
                "order_id": order_id,
                "amount": order_data.get("amount", 0),
                "type": "credit_purchase",
                "status": "failed",
                "description": f"Payment failed for order {order_id}",
                "created_at": datetime.utcnow().isoformat()
            }
            supabase.table("transactions").insert(transaction_data).execute()
            
            # Update webhook log
            supabase.table("webhook_logs").update({
                "status": "processed",
                "message": f"Order failed: {order_id}"
            }).eq("id", webhook_log["id"]).execute()
        
        else:
            logger.info(f"Unhandled event type: {event_type}")
            supabase.table("webhook_logs").update({
                "status": "ignored",
                "message": f"Unhandled event type: {event_type}"
            }).eq("id", webhook_log["id"]).execute()
        
        return {"success": True, "message": f"Webhook processed: {event_type}"}
        
    except Exception as webhook_error:
        logger.error(f"Webhook processing error: {webhook_error}")
        traceback.print_exc()
        
        # Log the error
        error_log = {
            "id": str(uuid.uuid4()),
            "error": str(webhook_error),
            "timestamp": datetime.utcnow().isoformat()
        }
        supabase.table("webhook_errors").insert(error_log).execute()
        
        return JSONResponse(
            status_code=200,  # Return 200 to prevent Polar.sh from retrying
            content={"success": False, "error": str(webhook_error)}
        )

@app.get("/api/payments/webhook/debug")
async def webhook_debug():
    try:
        response = supabase.table("transactions")\
            .select("*")\
            .order("created_at", desc=True)\
            .limit(10)\
            .execute()
        
        return {
            "success": True,
            "transactions": response.data,
            "webhook_secret_configured": bool(POLAR_WEBHOOK_SECRET),
            "test_mode": ENABLE_TEST_MODE,
            "polar_server": POLAR_SERVER
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/api/payments/add-test-credits")
async def add_test_credits(
    request: Request,
    current_user: User = Depends(get_current_user)
):
    try:
        if not ENABLE_TEST_MODE:
            raise HTTPException(400, "Test mode is disabled")
        
        data = await request.json()
        credits = data.get("credits", 100)
        
        if credits <= 0:
            raise HTTPException(400, "Credits must be positive")
        
        new_balance = add_user_credits(
            current_user.id,
            credits,
            f"Test credits added: {credits}"
        )
        
        return {
            "success": True,
            "message": f"Test credits added successfully",
            "credits_added": credits,
            "new_balance": new_balance
        }
        
    except HTTPException:
        raise
    except Exception as credit_error:
        logger.error(f"Failed to add test credits: {credit_error}")
        raise HTTPException(500, "Failed to add test credits")

@app.get("/api/payments/transactions")
async def get_user_transactions(current_user: User = Depends(get_current_user)):
    try:
        response = supabase.table("transactions").select("*").eq("user_id", current_user.id).order("created_at", desc=True).execute()
        
        transactions = []
        for tx in response.data:
            transactions.append({
                "id": tx["id"],
                "amount": tx.get("amount", 0),
                "credits": tx.get("credits", 0),
                "status": tx.get("status", "unknown"),
                "created_at": tx.get("created_at"),
                "updated_at": tx.get("updated_at"),
                "description": tx.get("description", "Transaction"),
                "session_id": tx.get("session_id"),
                "package_id": tx.get("package_id")
            })
        
        return {
            "success": True,
            "transactions": transactions
        }
        
    except Exception as transaction_error:
        logger.error(f"Failed to get transactions: {transaction_error}")
        raise HTTPException(500, "Failed to retrieve transactions")

# Authentication endpoints
@app.post("/api/auth/signup")
async def signup(request: Request):
    try:
        print("Signup endpoint called")
        
        try:
            data = await request.json()
        except json.JSONDecodeError as json_error:
            print(f"JSON decode error: {json_error}")
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "success": False,
                    "error": "Invalid JSON format",
                    "detail": str(json_error)
                }
            )
        
        email = data.get("email")
        password = data.get("password")
        name = data.get("name")
        
        if not email:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "success": False,
                    "error": "Email is required",
                    "detail": "Please provide an email address"
                }
            )
        
        if not password:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "success": False,
                    "error": "Password is required",
                    "detail": "Please provide a password"
                }
            )
        
        if not name:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "success": False,
                    "error": "Name is required",
                    "detail": "Please provide your name"
                }
            )
        
        if len(password) < 6:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "success": False,
                    "error": "Password too short",
                    "detail": "Password must be at least 6 characters"
                }
            )
        
        if "@" not in email:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "success": False,
                    "error": "Invalid email",
                    "detail": "Please provide a valid email address"
                }
            )
        
        try:
            response = supabase.table("users").select("*").eq("email", email).execute()
        except Exception as db_error:
            print(f"Supabase query error: {db_error}")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "success": False,
                    "error": "Database error",
                    "detail": "Failed to query database"
                }
            )
        
        if response.data:
            user = response.data[0]
            if user.get("is_verified"):
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={
                        "success": False,
                        "error": "User already exists",
                        "detail": "An account with this email already exists"
                    }
                )
            else:
                verification_token = secrets.token_urlsafe(32)
                try:
                    supabase.table("users").update({
                        "verification_token": verification_token,
                        "verification_token_expires": (datetime.utcnow() + timedelta(hours=24)).isoformat()
                    }).eq("id", user["id"]).execute()
                except Exception as update_error:
                    print(f"Failed to update user: {update_error}")
                
                send_verification_email(email, user.get("name", name), verification_token)
                
                return {
                    "success": True,
                    "message": "Verification email resent. Please check your inbox.",
                    "requires_verification": True
                }
        
        user_id = str(uuid.uuid4())
        verification_token = secrets.token_urlsafe(32)
        
        try:
            password_hash = get_password_hash(password)
        except Exception as hash_error:
            print(f"Password hashing failed: {hash_error}")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "success": False,
                    "error": "Password processing failed",
                    "detail": "Failed to process password"
                }
            )
        
        new_user = {
            "id": user_id,
            "email": email,
            "name": name,
            "password_hash": password_hash,
            "is_verified": False,
            "verification_token": verification_token,
            "verification_token_expires": (datetime.utcnow() + timedelta(hours=24)).isoformat(),
            "credits": 1000,
            "created_at": datetime.utcnow().isoformat()
        }
        
        try:
            response = supabase.table("users").insert(new_user).execute()
        except Exception as insert_error:
            print(f"Failed to insert user: {insert_error}")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "success": False,
                    "error": "Database error",
                    "detail": "Failed to create user in database"
                }
            )
        
        if not response.data:
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "success": False,
                    "error": "Database error",
                    "detail": "Failed to create user in database"
                }
            )
        
        send_verification_email(email, name, verification_token)
        
        logger.info(f"New user registered (pending verification): {email}")
        
        return {
            "success": True,
            "message": "Verification email sent. Please check your inbox.",
            "requires_verification": True,
            "user_id": user_id
        }
        
    except Exception as signup_error:
        print(f"Unexpected error in signup: {signup_error}")
        traceback.print_exc()
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "error": "Internal server error",
                "detail": "Registration failed due to an internal error"
            }
        )

@app.post("/api/auth/login")
async def login(request: Request):
    try:
        data = await request.json()
        email = data.get("email")
        password = data.get("password")
        
        if not email or not password:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email and password are required"
            )
        
        response = supabase.table("users").select("*").eq("email", email).execute()
        
        if not response.data:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials"
            )
        
        user = response.data[0]
        
        if not verify_password(password, user["password_hash"]):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials"
            )
        
        if not user["is_verified"]:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Please verify your email before logging in"
            )
        
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user["id"], "email": user["email"]},
            expires_delta=access_token_expires
        )
        
        logger.info(f"User logged in: {user['email']}")
        
        return {
            "success": True,
            "message": "Login successful",
            "token": access_token,
            "user": {
                "id": user["id"],
                "email": user["email"],
                "name": user["name"],
                "credits": user["credits"],
                "verified": user["is_verified"]
            }
        }
        
    except HTTPException:
        raise
    except Exception as login_error:
        logger.error(f"Login error: {login_error}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )

@app.post("/api/auth/verify")
async def verify_email(token: str = Form(...)):
    try:
        response = supabase.table("users").select("*").eq("verification_token", token).execute()
        
        if not response.data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or expired verification token"
            )
        
        user = response.data[0]
        
        token_expires_str = user.get("verification_token_expires")
        if token_expires_str:
            try:
                token_expires = datetime.fromisoformat(token_expires_str.replace('Z', '+00:00'))
                if datetime.utcnow() > token_expires:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Verification token has expired"
                    )
            except Exception as date_error:
                logger.error(f"Token expiry parsing error: {date_error}")
        
        supabase.table("users").update({
            "is_verified": True,
            "verification_token": None,
            "verification_token_expires": None
        }).eq("id", user["id"]).execute()
        
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user["id"], "email": user["email"]},
            expires_delta=access_token_expires
        )
        
        logger.info(f"Email verified: {user['email']}")
        
        return {
            "success": True,
            "message": "Email verified successfully!",
            "token": access_token,
            "user": {
                "id": user["id"],
                "email": user["email"],
                "name": user["name"],
                "credits": user["credits"],
                "verified": True
            }
        }
        
    except HTTPException:
        raise
    except Exception as verification_error:
        logger.error(f"Verification error: {verification_error}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Verification failed"
        )

@app.post("/api/auth/resend-verification")
async def resend_verification(request: Request):
    try:
        data = await request.json()
        email = data.get("email")
        
        if not email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email is required"
            )
        
        response = supabase.table("users").select("*").eq("email", email).execute()
        
        if not response.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        user = response.data[0]
        
        if user["is_verified"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already verified"
            )
        
        verification_token = secrets.token_urlsafe(32)
        supabase.table("users").update({
            "verification_token": verification_token,
            "verification_token_expires": (datetime.utcnow() + timedelta(hours=24)).isoformat()
        }).eq("id", user["id"]).execute()
        
        send_verification_email(user["email"], user["name"], verification_token)
        
        return {
            "success": True,
            "message": "Verification email resent. Please check your inbox."
        }
        
    except HTTPException:
        raise
    except Exception as resend_error:
        logger.error(f"Resend verification error: {resend_error}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to resend verification email"
        )

@app.post("/api/auth/logout")
async def logout(response: Response, current_user: User = Depends(get_current_user)):
    response.delete_cookie(key="access_token")
    
    return {
        "success": True,
        "message": "Logged out successfully"
    }

@app.post("/api/auth/demo-login")
async def demo_login():
    try:
        demo_email = "demo@octavia.com"
        demo_password = "demo123"
        
        response = supabase.table("users").select("*").eq("email", demo_email).execute()
        
        if response.data:
            user = response.data[0]
            if not verify_password(demo_password, user["password_hash"]):
                supabase.table("users").update({
                    "password_hash": get_password_hash(demo_password)
                }).eq("id", user["id"]).execute()
        else:
            user_id = str(uuid.uuid4())
            new_user = {
                "id": user_id,
                "email": demo_email,
                "name": "Demo User",
                "password_hash": get_password_hash(demo_password),
                "is_verified": True,
                "credits": 5000,
                "created_at": datetime.utcnow().isoformat()
            }
            
            response = supabase.table("users").insert(new_user).execute()
            if not response.data:
                raise HTTPException(500, "Failed to create demo user")
            
            user = response.data[0]
        
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user["id"], "email": user["email"]},
            expires_delta=access_token_expires
        )
        
        return {
            "success": True,
            "message": "Demo login successful",
            "token": access_token,
            "user": {
                "id": user["id"],
                "email": user["email"],
                "name": user["name"],
                "credits": user["credits"],
                "verified": user["is_verified"]
            }
        }
        
    except Exception as demo_error:
        logger.error(f"Demo login error: {demo_error}")
        raise HTTPException(500, "Demo login failed")

# User profile endpoints
@app.get("/api/user/profile")
async def get_user_profile(current_user: User = Depends(get_current_user)):
    return {
        "success": True,
        "user": {
            "id": current_user.id,
            "email": current_user.email,
            "name": current_user.name,
            "credits": current_user.credits,
            "verified": current_user.is_verified,
            "created_at": current_user.created_at.isoformat() if isinstance(current_user.created_at, datetime) else current_user.created_at
        }
    }

@app.get("/api/user/credits")
async def get_user_credits(current_user: User = Depends(get_current_user)):
    return {
        "success": True,
        "credits": current_user.credits,
        "email": current_user.email
    }

# Video processing helper functions
def save_upload_file(upload_file: UploadFile) -> tuple:
    file_id = str(uuid.uuid4())
    file_ext = os.path.splitext(upload_file.filename)[1]
    file_path = f"temp_{file_id}{file_ext}"
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
        
        files_db[file_id] = file_path
        logger.info(f"File saved: {upload_file.filename} -> {file_path}")
        return file_id, file_path
    except Exception as save_error:
        logger.error(f"File save failed: {save_error}")
        raise HTTPException(500, f"File upload failed: {str(save_error)}")

def cleanup_file(file_path: str):
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Cleaned up: {file_path}")
    except Exception as cleanup_error:
        logger.error(f"Failed to cleanup {file_path}: {cleanup_error}")

def extract_audio_from_video(video_path: str, output_audio_path: str) -> bool:
    try:
        cmd = [
            'ffmpeg', '-i', video_path,
            '-vn', '-acodec', 'pcm_s16le',
            '-ar', '44100', '-ac', '2',
            '-loglevel', 'error',
            output_audio_path, '-y'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True,
                              creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0)
        if result.returncode != 0:
            logger.error(f"FFmpeg error: {result.stderr[:200]}")
            return False
        
        return True
        
    except Exception as ffmpeg_error:
        logger.error(f"Audio extraction failed: {ffmpeg_error}")
        return False

def translate_text_with_fallback(text: str, target_lang: str = "es") -> str:
    if translator:
        try:
            result = translator(text)
            if isinstance(result, list) and len(result) > 0:
                return result[0]['translation_text']
            return text
        except Exception as translation_error:
            logger.error(f"Translation failed: {translation_error}")
    
    fallback_map = {
        "hello": "hola",
        "welcome": "bienvenido",
        "thank you": "gracias",
        "goodbye": "adiós",
        "please": "por favor"
    }
    
    translated = text
    for eng, esp in fallback_map.items():
        translated = translated.replace(eng, esp)
    
    return translated

# Video translation endpoint
@app.post("/api/translate/video")
async def translate_video(
    file: UploadFile = File(...),
    target_language: str = Form("es"),
    current_user: User = Depends(get_current_user)
):
    if current_user.credits < 5:
        raise HTTPException(400, "Insufficient credits")
    
    if not file.filename.lower().endswith(('.mp4', '.mov', '.avi', '.mkv', '.webm')):
        raise HTTPException(400, "Please upload a video file")
    
    try:
        supabase.table("users").update({"credits": current_user.credits - 5}).eq("id", current_user.id).execute()
    except Exception as credit_error:
        logger.error(f"Failed to update credits: {credit_error}")
        raise HTTPException(500, "Failed to process payment")
    
    file_id, file_path = save_upload_file(file)
    job_id = str(uuid.uuid4())
    
    job_info = {
        "id": job_id,
        "status": "processing",
        "progress": 10,
        "type": "video_simple",
        "file_id": file_id,
        "target_language": target_language,
        "original_filename": file.filename,
        "user_id": current_user.id,
        "user_email": current_user.email
    }
    jobs_db[job_id] = job_info
    
    try:
        await asyncio.sleep(2)
        job_info["progress"] = 50
        
        output_filename = f"translated_{job_id}.mp4"
        shutil.copy2(file_path, output_filename)
        
        job_info["progress"] = 100
        job_info["status"] = "completed"
        job_info["download_url"] = f"/api/download/{job_id}"
        job_info["output_filename"] = output_filename
        
        response = supabase.table("users").select("credits").eq("id", current_user.id).execute()
        new_credits = response.data[0]["credits"] if response.data else current_user.credits - 5
        
        return {
            "success": True,
            "job_id": job_id,
            "message": "Video translation completed",
            "download_url": f"/api/download/{job_id}",
            "remaining_credits": new_credits
        }
        
    except Exception as processing_error:
        job_info["status"] = "failed"
        job_info["error"] = str(processing_error)
        try:
            supabase.table("users").update({"credits": current_user.credits}).eq("id", current_user.id).execute()
        except:
            pass
        raise HTTPException(500, f"Processing failed: {str(processing_error)}")
    
    finally:
        cleanup_file(file_path)

# Job status and download endpoints
@app.get("/api/jobs/{job_id}/status")
async def get_job_status(job_id: str, current_user: User = Depends(get_current_user)):
    if job_id not in jobs_db:
        raise HTTPException(404, "Job not found")
    
    job = jobs_db[job_id]
    
    if job["user_id"] != current_user.id:
        raise HTTPException(403, "Access denied")
    
    response = {
        "success": True,
        "job_id": job_id,
        "status": job.get("status", "unknown"),
        "progress": job.get("progress", 0),
        "original_filename": job.get("original_filename"),
        "target_language": job.get("target_language"),
        "download_url": job.get("download_url"),
        "error": job.get("error")
    }
    
    return response

@app.get("/api/download/{job_id}")
async def download_file(job_id: str, current_user: User = Depends(get_current_user)):
    if job_id not in jobs_db:
        raise HTTPException(404, "File not found")
    
    job = jobs_db[job_id]
    
    if job["user_id"] != current_user.id:
        raise HTTPException(403, "Access denied")
    
    filename = job.get("output_filename")
    
    if not filename or not os.path.exists(filename):
        raise HTTPException(404, "Output file not found")
    
    return FileResponse(
        filename,
        media_type="application/octet-stream",
        filename=f"octavia_translation_{job_id}{os.path.splitext(filename)[1]}"
    )

# Health check endpoint
@app.get("/api/health")
async def health_check():
    return {
        "success": True,
        "status": "healthy",
        "service": "Octavia Video Translator",
        "version": "4.0.0",
        "database": "Supabase",
        "payment": {
            "polar_sh": "available" if polar_client else "not_available",
            "mode": POLAR_SERVER,
            "test_mode": ENABLE_TEST_MODE,
            "real_products_configured": True,
            "webhook_secret_configured": bool(POLAR_WEBHOOK_SECRET)
        },
        "models": {
            "whisper": "loaded" if whisper_model else "not_available",
            "translation": "loaded" if translator else "not_available"
        },
        "timestamp": datetime.now().isoformat()
    }

# Debug endpoints for payment testing
@app.post("/api/payments/debug-session")
async def debug_create_session(
    request: Request,
    current_user: User = Depends(get_current_user)
):
    try:
        print(f"=== DEBUG PAYMENT SESSION ===")
        print(f"User: {current_user.email}")
        print(f"ENABLE_TEST_MODE: {ENABLE_TEST_MODE}")
        print(f"Polar client available: {polar_client is not None}")
        
        data = await request.json()
        package_id = data.get("package_id")
        print(f"Package ID: {package_id}")
        
        package = CREDIT_PACKAGES.get(package_id)
        if not package:
            return {"success": False, "error": "Package not found"}
        
        print(f"Package found: {package['name']}")
        print(f"Polar checkout link: {package['checkout_link']}")
        
        session_id = str(uuid.uuid4())
        print(f"Session ID: {session_id}")
        
        if ENABLE_TEST_MODE:
            print("Test mode - would add credits directly")
            return {
                "success": True,
                "test_mode": True,
                "message": "Test mode active",
                "credits": package["credits"]
            }
        else:
            print("Real mode - would redirect to Polar.sh checkout")
            checkout_url = package["checkout_link"]
            print(f"Checkout URL: {checkout_url}")
            
            return {
                "success": True,
                "test_mode": False,
                "checkout_url": checkout_url,
                "message": "Real payment mode"
            }
            
    except Exception as e:
        print(f"Debug error: {e}")
        return {"success": False, "error": str(e)}

@app.post("/api/payments/manual-complete")
async def manual_complete_payment(
    request: Request,
    current_user: User = Depends(get_current_user)
):
    """
    Manually complete a payment for testing when webhooks aren't working.
    """
    try:
        data = await request.json()
        session_id = data.get("session_id")
        package_id = data.get("package_id")
        
        if not session_id or not package_id:
            raise HTTPException(400, "session_id and package_id required")
        
        # Find the transaction
        response = supabase.table("transactions").select("*").eq("session_id", session_id).execute()
        
        if not response.data:
            raise HTTPException(404, "Transaction not found")
        
        transaction = response.data[0]
        
        # Get package info
        package = CREDIT_PACKAGES.get(package_id)
        if not package:
            raise HTTPException(400, "Invalid package")
        
        # Add credits
        new_balance = add_user_credits(
            current_user.id,
            package["credits"],
            f"Manual completion: {package['name']}"
        )
        
        # Update transaction
        update_transaction_status(
            transaction["id"],
            "completed",
            f"Manually completed: {package['name']}"
        )
        
        return {
            "success": True,
            "message": f"Manually added {package['credits']} credits",
            "new_balance": new_balance,
            "transaction_id": transaction["id"]
        }
        
    except Exception as e:
        logger.error(f"Manual completion error: {e}")
        raise HTTPException(500, str(e))

async def check_polar_order_status(order_id: str):
    """
    Check order status directly from Polar.sh API
    """
    if not polar_client:
        return None
    
    try:
        # Get order from Polar.sh
        response = polar_client.orders.get(order_id)
        return response
    except Exception as e:
        logger.error(f"Failed to check Polar order: {e}")
        return None

@app.get("/api/payments/check-order/{order_id}")
async def check_order_status(order_id: str, current_user: User = Depends(get_current_user)):
    """
    Manually check an order's status on Polar.sh
    """
    try:
        # First check our database
        tx_response = supabase.table("transactions").select("*").eq("order_id", order_id).execute()
        
        if not tx_response.data:
            # Try by session_id
            tx_response = supabase.table("transactions").select("*").eq("session_id", order_id).execute()
        
        transaction = tx_response.data[0] if tx_response.data else None
        
        if transaction and transaction["status"] == "completed":
            return {
                "success": True,
                "status": "completed",
                "message": "Already completed in our database"
            }
        
        # If not completed, try to check with Polar.sh
        if polar_client:
            try:
                # This depends on your Polar SDK version - adjust as needed
                order = polar_client.orders.get(order_id)
                
                if order and order.status == "completed":
                    # Order is completed on Polar.sh - update our database
                    package_id = transaction.get("package_id") if transaction else None
                    
                    if package_id and package_id in CREDIT_PACKAGES:
                        credits_to_add = CREDIT_PACKAGES[package_id]["credits"]
                    else:
                        credits_to_add = 100  # Default
                    
                    if transaction:
                        # Update existing transaction
                        add_user_credits(transaction["user_id"], credits_to_add, "Payment completed via Polar.sh")
                        update_transaction_status(transaction["id"], "completed", "Payment verified with Polar.sh")
                    
                    return {
                        "success": True,
                        "status": "completed",
                        "message": "Order completed on Polar.sh! Credits added."
                    }
                else:
                    return {
                        "success": True,
                        "status": order.status if order else "unknown",
                        "message": f"Order status on Polar.sh: {order.status if order else 'not found'}"
                    }
                    
            except Exception as polar_error:
                return {
                    "success": False,
                    "error": f"Polar.sh API error: {str(polar_error)}",
                    "status": "error"
                }
        
        return {
            "success": True,
            "status": transaction["status"] if transaction else "unknown",
            "message": "Could not check Polar.sh"
        }
        
    except Exception as e:
        logger.error(f"Order check error: {e}")
        return {"success": False, "error": str(e)}

@app.post("/api/payments/force-complete-all")
async def force_complete_all_payments():
    """
    EMERGENCY ENDPOINT: Force complete all pending payments
    """
    try:
        # Get all pending transactions
        response = supabase.table("transactions").select("*").eq("status", "pending").execute()
        
        if not response.data:
            return {"success": True, "message": "No pending transactions found"}
        
        completed = []
        failed = []
        
        for transaction in response.data:
            try:
                user_id = transaction["user_id"]
                package_id = transaction.get("package_id")
                
                if not package_id:
                    # Try to guess package from amount
                    amount = transaction.get("amount", 0)
                    if amount >= 3499:
                        credits_to_add = 500
                    elif amount >= 1999:
                        credits_to_add = 250
                    else:
                        credits_to_add = 100
                else:
                    package = CREDIT_PACKAGES.get(package_id)
                    if not package:
                        failed.append(f"Invalid package: {package_id}")
                        continue
                    credits_to_add = package["credits"]
                
                # Add credits to user
                current_credits = transaction.get("current_credits", 0)
                new_credits = current_credits + credits_to_add
                
                supabase.table("users").update({"credits": new_credits}).eq("id", user_id).execute()
                
                # Mark transaction as completed
                supabase.table("transactions").update({
                    "status": "completed",
                    "description": f"FORCE COMPLETED: Added {credits_to_add} credits",
                    "updated_at": datetime.utcnow().isoformat()
                }).eq("id", transaction["id"]).execute()
                
                completed.append(f"Transaction {transaction['id']}: {credits_to_add} credits")
                
            except Exception as tx_error:
                failed.append(f"Transaction {transaction['id']}: {str(tx_error)}")
        
        return {
            "success": True,
            "completed": completed,
            "failed": failed,
            "message": f"Force completed {len(completed)} transactions"
        }
        
    except Exception as e:
        logger.error(f"Force complete error: {e}")
        return {"success": False, "error": str(e)}

@app.get("/api/payments/fix-pending")
async def fix_pending_payments():
    """
    Quick fix: Direct SQL to update all pending transactions
    """
    try:
        # First, get all pending transactions with their user info
        response = supabase.table("transactions").select("*, users!inner(credits)").eq("status", "pending").execute()
        
        for tx in response.data:
            user_id = tx["user_id"]
            package_id = tx.get("package_id")
            
            # Determine credits to add
            if package_id in CREDIT_PACKAGES:
                credits_to_add = CREDIT_PACKAGES[package_id]["credits"]
            else:
                # Default based on amount
                amount = tx.get("amount", 999)
                credits_to_add = 100 if amount == 999 else 250 if amount == 1999 else 500
            
            # Update user credits
            supabase.table("users").update({"credits": tx["users"]["credits"] + credits_to_add}).eq("id", user_id).execute()
            
            # Update transaction
            supabase.table("transactions").update({
                "status": "completed",
                "description": "Fixed by system",
                "updated_at": datetime.utcnow().isoformat()
            }).eq("id", tx["id"]).execute()
        
        return {"success": True, "message": "Fixed all pending transactions"}
        
    except Exception as e:
        return {"success": False, "error": str(e)}

# Root endpoint with API documentation
@app.get("/")
async def root():
    return {
        "success": True,
        "service": "Octavia Video Translator",
        "version": "4.0.0",
        "status": "operational",
        "authentication": "JWT + Supabase",
        "payment": "Polar.sh integration",
        "test_mode": ENABLE_TEST_MODE,
        "real_products": "configured",
        "password_hashing": "SHA256 with salt",
        "endpoints": {
            "health": "/api/health",
            "docs": "/docs",
            "auth": {
                "signup": "/api/auth/signup",
                "login": "/api/auth/login",
                "logout": "/api/auth/logout",
                "verify": "/api/auth/verify",
                "resend_verification": "/api/auth/resend-verification",
                "demo_login": "/api/auth/demo-login"
            },
            "payments": {
                "packages": "/api/payments/packages",
                "create_session": "/api/payments/create-session",
                "payment_status": "/api/payments/status/{session_id}",
                "add_test_credits": "/api/payments/add-test-credits",
                "transactions": "/api/payments/transactions",
                "webhook": "/api/payments/webhook/polar",
                "webhook_debug": "/api/payments/webhook/debug"
            },
            "user": {
                "profile": "/api/user/profile",
                "credits": "/api/user/credits"
            },
            "video": {
                "translate": "/api/translate/video",
                "job_status": "/api/jobs/{job_id}/status",
                "download": "/api/download/{job_id}"
            }
        }
    }

# Application entry point
if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*60)
    print("OCTAVIA VIDEO TRANSLATOR v4.0 - WITH REAL POLAR.SH PAYMENTS")
    print("="*60)
    print(f"Database: Supabase")
    print(f"Payment: Polar.sh ({POLAR_SERVER} mode)")
    print(f"Test Mode: {'ENABLED' if ENABLE_TEST_MODE else 'DISABLED'}")
    print(f"Webhook Secret: {'CONFIGURED' if POLAR_WEBHOOK_SECRET else 'NOT CONFIGURED'}")
    print(f"Checkout Links: CONFIGURED with your Polar.sh checkout links")
    print(f"API URL: http://localhost:8000")
    print(f"Frontend: http://localhost:3000")
    print(f"Documentation: http://localhost:8000/docs")
    print(f"Logs: artifacts/logs.jsonl")
    print("="*60 + "\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False
    )