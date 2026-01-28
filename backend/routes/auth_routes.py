from fastapi import APIRouter, Request, HTTPException, Form, status, Depends, Response
from fastapi.responses import JSONResponse
from datetime import datetime, timedelta
import secrets
import json
import logging
import traceback
import uuid

from shared_dependencies import (
    supabase, User, verify_password, get_password_hash,
    create_access_token, get_current_user, get_current_user_id, password_manager
)
from config import DEMO_MODE
from utils import send_verification_email
from exceptions import *
from typing import Dict, Any

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auth", tags=["authentication"])

ACCESS_TOKEN_EXPIRE_MINUTES = 30

@router.post("/signup")
async def signup(request: Request):
    try:
        print("Signup endpoint called")

        try:
            data = await request.json()
        except json.JSONDecodeError as json_error:
            print(f"JSON decode error: {json_error}")
            raise ValidationError("Invalid JSON format", "request")

        email = data.get("email")
        password = data.get("password")
        name = data.get("name")

        if not email:
            raise ValidationError("Email is required", "email")

        if not password:
            raise ValidationError("Password is required", "password")

        if not name:
            raise ValidationError("Name is required", "name")

        if len(password) < 6:
            raise ValidationError("Password must be at least 6 characters", "password")

        if "@" not in email:
            raise ValidationError("Please provide a valid email address", "email")

        try:
            response = supabase.table("users").select("*").eq("email", email).execute()
        except Exception as db_error:
            print(f"Supabase query error: {db_error}")
            raise JobProcessingError("", "Database error: Failed to query database")

        if response.data:
            user = response.data[0]
            if user.get("is_verified"):
                raise ValidationError("An account with this email already exists", "email")
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
            raise JobProcessingError("", "Password processing failed")

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
            raise JobProcessingError("", "Database error: Failed to create user")

        if not response.data:
            raise JobProcessingError("", "Database error: Failed to create user")

        send_verification_email(email, name, verification_token)

        logger.info(f"New user registered (pending verification): {email}")

        return {
            "success": True,
            "message": "Verification email sent. Please check your inbox.",
            "requires_verification": True,
            "user_id": user_id
        }

    except OctaviaException:
        raise
    except Exception as signup_error:
        print(f"Unexpected error in signup: {signup_error}")
        traceback.print_exc()
        raise JobProcessingError("", "Registration failed due to an internal error")

@router.post("/login")
async def login(request: Request):
    try:
        data = await request.json()
        email = data.get("email")
        password = data.get("password")

        if not email or not password:
            raise ValidationError("Email and password are required")

        # Check for demo mode login (only works when DEMO_MODE=true)
        if DEMO_MODE and email == "demo@octavia.com" and password == "demo123":
            logger.info(f"Demo user logged in")
            access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
            access_token = create_access_token(
                data={"sub": "demo-user-id", "email": "demo@octavia.com"},
                expires_delta=access_token_expires
            )
            return {
                "success": True,
                "message": "Login successful",
                "token": access_token,
                "user": {
                    "id": "demo-user-id",
                    "email": "demo@octavia.com",
                    "name": "Demo User",
                    "credits": 5000,
                    "verified": True
                }
            }

        # Regular login requires Supabase to be configured
        if supabase is None:
            raise AuthenticationError("Database not configured. Please set up Supabase or use demo mode.")

        response = supabase.table("users").select("*").eq("email", email).execute()

        if not response.data:
            raise AuthenticationError("Invalid credentials")

        user = response.data[0]

        if not verify_password(password, user["password_hash"]):
            raise AuthenticationError("Invalid credentials")

        if not user["is_verified"]:
            raise AuthenticationError("Please verify your email before logging in")

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

    except OctaviaException:
        raise
    except Exception as login_error:
        logger.error(f"Login error: {login_error}")
        raise JobProcessingError("", "Login failed")

@router.post("/verify")
async def verify_email(token: str = Form(...)):
    try:
        response = supabase.table("users").select("*").eq("verification_token", token).execute()

        if not response.data:
            raise ValidationError("Invalid or expired verification token", "token")

        user = response.data[0]

        token_expires_str = user.get("verification_token_expires")
        if token_expires_str:
            try:
                token_expires = datetime.fromisoformat(token_expires_str.replace('Z', '+00:00'))
                if datetime.utcnow() > token_expires:
                    raise ValidationError("Verification token has expired", "token")
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

    except OctaviaException:
        raise
    except Exception as verification_error:
        logger.error(f"Verification error: {verification_error}")
        raise JobProcessingError("", "Verification failed")

@router.post("/resend-verification")
async def resend_verification(request: Request):
    try:
        data = await request.json()
        email = data.get("email")

        if not email:
            raise ValidationError("Email is required", "email")

        response = supabase.table("users").select("*").eq("email", email).execute()

        if not response.data:
            raise NotFoundError("User", email)

        user = response.data[0]

        if user["is_verified"]:
            raise ValidationError("Email already verified", "email")

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

    except OctaviaException:
        raise
    except Exception as resend_error:
        logger.error(f"Resend verification error: {resend_error}")
        raise JobProcessingError("", "Failed to resend verification email")

@router.post("/logout")
async def logout(response: Response, current_user: User = Depends(get_current_user)):
    response.delete_cookie(key="access_token")

    return {
        "success": True,
        "message": "Logged out successfully"
    }

@router.post("/demo-login")
async def demo_login():
    """Demo login endpoint - simplified version"""
    print("Demo login called - simplified version")

    # Simple hardcoded response for testing
    return JSONResponse(content={
        "success": True,
        "message": "Demo login successful",
        "token": "demo_token_12345",
        "user": {
            "id": "550e8400-e29b-41d4-a716-446655440000",
            "email": "demo@octavia.com",
            "name": "Demo User",
            "credits": 5000,
            "verified": True
        }
    })
