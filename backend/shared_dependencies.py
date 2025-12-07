"""
Shared dependencies and utilities for the Octavia backend
"""
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
import secrets
import hashlib
from datetime import datetime, timedelta
from pydantic import BaseModel, EmailStr
from typing import Optional
from supabase import create_client, Client
import os

# Configuration from environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

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

async def get_current_user(token: str = Depends(HTTPBearer())):
    """Get current user from JWT token"""
    payload = verify_token(token.credentials)
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
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error fetching user data",
        )
