"""
Shared dependencies and utilities for the Octavia backend
"""
import os
from dotenv import load_dotenv, find_dotenv

# Load environment variables early - look for .env in current or parent dirs
load_dotenv(find_dotenv(), override=True)

from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
import secrets
import hashlib
from datetime import datetime, timedelta
from pydantic import BaseModel, EmailStr
from typing import Optional
from supabase import create_client, Client

# Configuration from environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Debug: Check if demo mode is set
DEMO_MODE_ENV = os.getenv("DEMO_MODE", "false")
print(f"DEBUG: DEMO_MODE environment variable: '{DEMO_MODE_ENV}'")
print(f"DEBUG: DEMO_MODE parsed: {DEMO_MODE_ENV.lower() == 'true'}")

# Initialize Supabase client
try:
    if SUPABASE_URL and SUPABASE_SERVICE_KEY:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    else:
        print("WARNING: SUPABASE_URL or SUPABASE_SERVICE_KEY not set. Supabase client not initialized.")
        supabase = None
except Exception as e:
    print(f"WARNING: Failed to initialize Supabase client: {e}")
    supabase = None

# Initialize Redis client
REDIS_URL = os.getenv("REDIS_URL")
redis_client = None

if REDIS_URL:
    try:
        import redis
        # Use from_url for easy connection string handling (including SSL)
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        # Test connection
        redis_client.ping()
        print("Successfully connected to Redis")
    except Exception as e:
        print(f"WARNING: Failed to initialize Redis client: {e}")
        redis_client = None
else:
    print("WARNING: REDIS_URL not set. Redis client not initialized.")

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

# Pydantic models for request/response validation
class User(BaseModel):
    id: str
    email: str
    name: str
    is_verified: bool
    credits: int
    created_at: datetime

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

async def get_current_user_id(credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())):
    """Get current user ID from JWT token without database lookup"""
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

    return user_id

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())):
    """Get current user from JWT token"""
    token = credentials.credentials

    # DEMO_MODE: check for demo tokens first
    DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() == "true"
    if DEMO_MODE and token.startswith("demo_token_"):
        # In demo mode, accept demo tokens
        print("DEBUG: Using demo user with demo token")
        return User(
            id="550e8400-e29b-41d4-a716-446655440000",
            email="demo@octavia.com",
            name="Demo User",
            is_verified=True,
            credits=5000,
            created_at=datetime.utcnow()
        )

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

    print(f"DEBUG: DEMO_MODE={DEMO_MODE}, user_id={user_id}")

    if DEMO_MODE:
        # In demo mode, return static demo user for any valid token
        print("DEBUG: Using demo user")
        return User(
            id=user_id,
            email="demo@octavia.com",
            name="Demo User",
            is_verified=True,
            credits=5000,
            created_at=datetime.utcnow()
        )

    # Normal mode: fetch user from Supabase
    try:
        from services.db_utils import with_retry
        
        async def fetch_user():
            return supabase.table("users").select("*").eq("id", user_id).execute()
            
        response = await with_retry(fetch_user)
        
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
    except HTTPException:
        raise
    except Exception as db_error:
        print(f"Database error after retries: {db_error}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error fetching user data from database",
        )
