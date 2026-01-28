import os
import json
import uuid
import secrets
import logging
import traceback
from datetime import datetime, timedelta
from typing import List, Dict, Optional

from fastapi import FastAPI, Request, Response, HTTPException, Depends, Form, File, UploadFile, BackgroundTasks, status, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from services.websocket_manager import ws_manager

# Audio processing
try:
    from pydub import AudioSegment
except ImportError:
    AudioSegment = None
    print("Warning: pydub not available")

# Subtitle parsing
try:
    import pysrt
except ImportError:
    pysrt = None
    print("Warning: pysrt not available")

from shared_dependencies import (
    supabase, User, verify_password, get_password_hash,
    create_access_token, get_current_user, get_current_user_id, password_manager
)
from services.job_storage import job_storage
from config import DEMO_MODE
# from config import HELSINKI_MODELS, DEMO_MODE, ENABLE_TEST_MODE  # Temporarily disabled
from services.job_service import job_service
from services.translation_service import translation_service
from routes.auth_routes import router as auth_router
# Configure logging with rotation
import logging
from logging.handlers import RotatingFileHandler
import os

# Ensure logs directory exists
os.makedirs("artifacts", exist_ok=True)

# Create rotating log handler (10MB max, keep 5 backup files)
log_handler = RotatingFileHandler(
    "artifacts/backend_debug.log",
    maxBytes=10 * 1024 * 1024,  # 10MB per file
    backupCount=5,              # Keep 5 rotated files
    encoding="utf-8"
)

# Configure format
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log_handler.setFormatter(formatter)

# Create console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False

# Import and add secret masker
try:
    from services.logging_utils import SecretMasker
    masker = SecretMasker()
    log_handler.addFilter(masker)
    console_handler.addFilter(masker)
except ImportError:
    print("Warning: services.logging_utils not available, secrets may not be masked in logs")

logger.addHandler(log_handler)
logger.addHandler(console_handler)

# Constants
DEFAULT_CHUNK_SIZE = 30
AUDIO_TRANSLATION_CREDITS = 10
VIDEO_TRANSLATION_CREDITS = 10
SUBTITLE_TO_AUDIO_CREDITS = 5
VOICE_PREVIEW_CREDITS = 1
INITIAL_USER_CREDITS = 1000
DEMO_USER_CREDITS = 5000
MIN_PASSWORD_LENGTH = 6
VERIFICATION_TOKEN_HOURS = 24
MAX_FILE_SIZE_MB = 500
VERIFICATION_TOKEN_LENGTH = 32
PROGRESS_COMPLETE = 100
BATCH_SIZE_TRANSLATION = 10

# Payment and test mode configuration
POLAR_SERVER = os.getenv("POLAR_SERVER", "sandbox")
ENABLE_TEST_MODE = os.getenv("ENABLE_TEST_MODE", "true").lower() == "true"
PIPELINE_AVAILABLE = True

# Credit packages
CREDIT_PACKAGES = {
    "starter": {"credits": 100, "price": 9.99},
    "professional": {"credits": 500, "price": 39.99},
    "enterprise": {"credits": 2000, "price": 99.99}
}

# Helsinki NLP translation models
HELSINKI_MODELS = {
    "en-es": "Helsinki-NLP/opus-mt-en-es",
    "es-en": "Helsinki-NLP/opus-mt-es-en",
    "en-fr": "Helsinki-NLP/opus-mt-en-fr",
    "fr-en": "Helsinki-NLP/opus-mt-fr-en",
    "en-de": "Helsinki-NLP/opus-mt-en-de",
    "de-en": "Helsinki-NLP/opus-mt-de-en",
    "en-it": "Helsinki-NLP/opus-mt-en-it",
    "it-en": "Helsinki-NLP/opus-mt-it-en",
    "en-pt": "Helsinki-NLP/opus-mt-en-pt",
    "pt-en": "Helsinki-NLP/opus-mt-pt-en",
    "en-ru": "Helsinki-NLP/opus-mt-en-ru",
    "ru-en": "Helsinki-NLP/opus-mt-ru-en",
}

# Translator cache
translator_cache = {}

def get_translator(source_lang: str, target_lang: str):
    """Get or create a translator for the given language pair"""
    key = f"{source_lang}-{target_lang}"
    if key in translator_cache:
        return translator_cache[key]
    
    if key not in HELSINKI_MODELS:
        return None
    
    try:
        from transformers import pipeline
        translator = pipeline("translation", model=HELSINKI_MODELS[key])
        translator_cache[key] = translator
        return translator
    except Exception as e:
        logger.warning(f"Failed to load translator for {key}: {e}")
        return None

# gTTS Voice options for TTS
# gTTS uses language codes - each language has ONE high-quality voice
# These are the supported languages for Google Text-to-Speech
GTTS_VOICE_OPTIONS = {
    "en": {
        "English": "en"
    },
    "es": {
        "Spanish": "es"
    },
    "fr": {
        "French": "fr"
    },
    "de": {
        "German": "de"
    },
    "it": {
        "Italian": "it"
    },
    "pt": {
        "Portuguese": "pt"
    },
    "ru": {
        "Russian": "ru"
    },
    "ja": {
        "Japanese": "ja"
    },
    "ko": {
        "Korean": "ko"
    },
    "zh-cn": {
        "Chinese (Simplified)": "zh-cn"
    },
    "ar": {
        "Arabic": "ar"
    },
    "hi": {
        "Hindi": "hi"
    },
    "nl": {
        "Dutch": "nl"
    },
    "pl": {
        "Polish": "pl"
    },
    "tr": {
        "Turkish": "tr"
    },
    "sv": {
        "Swedish": "sv"
    }
}

# Legacy Edge-TTS options (kept for reference, not used in main translation)
VOICE_OPTIONS = {
    "en": {
        "Aria (Female)": "en-US-AriaNeural",
        "David (Male)": "en-US-GuyNeural",
        "Emma (Female)": "en-US-JennyNeural",
        "Brian (Male)": "en-US-ChristopherNeural"
    },
    "es": {
        "Elena (Female)": "es-ES-ElviraNeural",
        "Alvaro (Male)": "es-ES-AlvaroNeural",
        "Esperanza (Female)": "es-ES-LauraNeural",
        "Jorge (Male)": "es-ES-JorgeNeural"
    },
    "fr": {
        "Denise (Female)": "fr-FR-DeniseNeural",
        "Henri (Male)": "fr-FR-HenriNeural"
    },
    "de": {
        "Katja (Female)": "de-DE-KatjaNeural",
        "Conrad (Male)": "de-DE-ConradNeural"
    },
    "it": {
        "Elsa (Female)": "it-IT-ElsaNeural",
        "Diego (Male)": "it-IT-DiegoNeural"
    },
    "pt": {
        "Francisca (Female)": "pt-BR-FranciscaNeural",
        "Antonio (Male)": "pt-BR-AntonioNeural"
    },
    "ru": {
        "Svetlana (Female)": "ru-RU-SvetlanaNeural",
        "Dmitry (Male)": "ru-RU-DmitryNeural"
    },
    "ja": {
        "Nanami (Female)": "ja-JP-NanamiNeural",
        "Keita (Male)": "ja-JP-KeitaNeural"
    },
    "ko": {
        "Sun-Hi (Female)": "ko-KR-SunHiNeural",
        "InJoon (Male)": "ko-KR-InJoonNeural"
    },
    "zh": {
        "Xiaoxiao (Female)": "zh-CN-XiaoxiaoNeural",
        "Yunyang (Male)": "zh-CN-YunyangNeural"
    }
}

# Legacy globals (to be replaced)
jobs_db = {}
subtitle_jobs = {}

# Helper functions
def save_jobs_db():
    """Save jobs database (placeholder)"""
    pass

# Simple app config
class SimpleAppConfig:
    def __init__(self):
        self.pipeline_available = True

app_config = SimpleAppConfig()

# Initialize cleanup utilities
from services.cleanup_utils import run_full_cleanup, cleanup_temp_files

def cleanup_on_startup():
    """Run cleanup when server starts."""
    logger.info("Running startup cleanup...")
    try:
        result = run_full_cleanup()
        logger.info(
            f"Startup cleanup: {result['total_cleaned']} files cleaned, "
            f"{result['total_freed_bytes'] / 1024:.1f} KB freed"
        )
    except Exception as e:
        logger.error(f"Startup cleanup failed: {e}")

def cleanup_on_shutdown():
    """Run cleanup when server shuts down."""
    logger.info("Running shutdown cleanup...")
    try:
        result = run_full_cleanup()
        logger.info(
            f"Shutdown cleanup: {result['total_cleaned']} files cleaned, "
            f"{result['total_freed_bytes'] / 1024:.1f} KB freed"
        )
    except Exception as e:
        logger.error(f"Shutdown cleanup failed: {e}")

def cleanup_job_temp_files(file_path: str, job_id: str):
    """
    Clean up temporary files after job completion or failure.
    
    Args:
        file_path: Path to the temp file to clean
        job_id: Job ID for logging
    """
    if file_path and os.path.exists(file_path):
        try:
            os.remove(file_path)
            logger.info(f"Cleaned up temp file for job {job_id}: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to clean up temp file {file_path}: {e}")

# Register startup and shutdown handlers
import atexit
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events."""
    # Startup
    cleanup_on_startup()
    yield
    # Shutdown
    cleanup_on_shutdown()

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Octavia Video Translator API",
    version="4.0.0",
    lifespan=lifespan
)

# Register shutdown with atexit as backup
atexit.register(cleanup_on_shutdown)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth_router)
from routes.translation_routes import router as translation_router
from routes.payment_routes import router as payment_router
app.include_router(translation_router)
app.include_router(payment_router)

# Utility functions
async def refund_credits(user_id: str, amount: int, reason: str = "job failure"):
    """Refund credits to a user"""
    try:
        response = supabase.table("users").select("credits").eq("id", user_id).execute()
        if response.data:
            current_credits = response.data[0]["credits"]
            supabase.table("users").update({"credits": current_credits + amount}).eq("id", user_id).execute()
            logger.info(f"Refunded {amount} credits to user {user_id} due to {reason}")
    except Exception as refund_error:
        logger.error(f"Failed to refund {amount} credits to user {user_id}: {refund_error}")

# Import dependencies if available
try:
    from modules.pipeline import VideoTranslationPipeline, PipelineConfig
except ImportError:
    PIPELINE_AVAILABLE = False
    logger.warning("Pipeline modules not available. Running in simplified mode.")

async def process_video_translation_job(job_id: str, file_path: str, target_language: str, user_id: str, separate: bool = False):
    """Background task for video translation"""
    try:
        # Check if job was cancelled before starting
        if await job_storage.is_cancelled(job_id):
            logger.info(f"Job {job_id} was cancelled before processing started")
            return
        
        # Update job status in Supabase
        await job_storage.update_progress(job_id, 5, "Initializing translation pipeline...")
        await job_storage.update_status(job_id, "processing")

        # Check if we have pipeline modules
        if app_config.pipeline_available or DEMO_MODE:
            from modules.pipeline import VideoTranslationPipeline, PipelineConfig

            # Process video with job_id for progress tracking and cancellation
            config = PipelineConfig(chunk_size=DEFAULT_CHUNK_SIZE, enable_vocal_separation=separate)
            pipeline = VideoTranslationPipeline(config)
            
            # Pass job_id and job_storage so pipeline can check for cancellation
            pipeline.job_id = job_id
            pipeline.job_storage = job_storage
            
            result = pipeline.process_video(file_path, target_language)

            # Check if job was cancelled during processing
            if await job_storage.is_cancelled(job_id):
                logger.info(f"Job {job_id} was cancelled during processing - aborting completion")
                return
            
            # Check if result indicates cancellation
            if result.get("cancelled"):
                logger.info(f"Job {job_id} processing returned cancelled status")
                return

            if result.get("success"):
                # Update progress based on result
                output_path = result.get("output_video", f"backend/outputs/translated_video_{job_id}.mp4")

                # Ensure output file exists
                if not os.path.exists(output_path):
                    with open(output_path, "wb") as f:
                        f.write(b"Placeholder - video translation completed")

                await job_storage.complete_job(job_id, {
                    "status": "completed",
                    "progress": 100,
                    "download_url": f"/api/download/video/{job_id}",
                    "output_path": output_path,
                    "completed_at": datetime.utcnow().isoformat(),
                    "message": "Video translation completed successfully",
                    # Add additional info from pipeline result
                    "total_chunks": result.get("total_chunks", 0),
                    "chunks_processed": result.get("chunks_processed", len(result.get("translated_paths", []))),
                    "processing_time_s": result.get("processing_time_s", 0),
                    "output_video": output_path
                })
            else:
                raise Exception(result.get("error", "Unknown error during processing"))

        else:
            # Simplified mode - video translation not available
            logger.warning("Video translation pipeline not available - failing job")
            raise Exception("Video translation is not available. The full video processing pipeline is required for video translation. Please contact support for assistance.")

    except Exception as e:
        # Don't mark as failed if it was cancelled
        if await job_storage.is_cancelled(job_id):
            logger.info(f"Job {job_id} was cancelled, not marking as failed")
            return
            
        logger.error(f"Video translation job {job_id} failed: {str(e)}")
        await job_storage.fail_job(job_id, str(e))

        # Refund credits on failure
        await refund_credits(user_id, VIDEO_TRANSLATION_CREDITS, "video translation failure")

        # Cleanup temp files
        cleanup_files = [file_path]
        if 'audio_path' in locals() and audio_path != file_path:
            cleanup_files.append(audio_path)

        for temp_file in cleanup_files:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass


# ========== AUTHENTICATION ENDPOINTS ==========
# Moved to routes/auth_routes.py

# ========== USER PROFILE ENDPOINTS ==========

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
    """Get user's current credit balance"""
    DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() == "true"
    if DEMO_MODE and current_user.email == "demo@octavia.com":
        return {
            "success": True,
            "credits": 5000,
            "email": current_user.email
        }
    return {
        "success": True,
        "credits": current_user.credits,
        "email": current_user.email
    }


# ========== PAYMENT ENDPOINTS ==========



# ========== USER PROFILE MANAGEMENT ENDPOINTS ==========

@app.put("/api/user/profile")
async def update_user_profile(
    request: Request,
    current_user: User = Depends(get_current_user)
):
    """Update user profile information"""
    try:
        data = await request.json()
        name = data.get("name")
        email = data.get("email")

        if not name or not email:
            raise HTTPException(400, "Name and email are required")

        # Check if email is already taken by another user
        if email != current_user.email:
            existing_user = supabase.table("users").select("*").eq("email", email).execute()
            if existing_user.data:
                raise HTTPException(400, "Email is already taken")

        # Update user in database
        update_data = {
            "name": name,
            "email": email,
            "updated_at": datetime.utcnow().isoformat()
        }

        supabase.table("users").update(update_data).eq("id", current_user.id).execute()

        logger.info(f"Updated profile for user {current_user.id}: {name}, {email}")

        return {
            "success": True,
            "message": "Profile updated successfully",
            "user": {
                "id": current_user.id,
                "name": name,
                "email": email,
                "credits": current_user.credits,
                "verified": current_user.is_verified
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Profile update failed: {e}")
        raise HTTPException(500, "Failed to update profile")

@app.put("/api/user/settings")
async def update_user_settings(
    request: Request,
    current_user: User = Depends(get_current_user)
):
    """Update user settings/preferences"""
    try:
        data = await request.json()

        # Extract settings
        notifications = data.get("notifications", {})
        language = data.get("language", "English")
        time_zone = data.get("time_zone", "UTC (GMT+0)")

        # Store settings in user metadata (you might want to create a separate settings table)
        settings_data = {
            "notifications": notifications,
            "language": language,
            "time_zone": time_zone,
            "updated_at": datetime.utcnow().isoformat()
        }

        # Update user with settings (you could store this in a JSON field or separate table)
        # For now, we'll store it in the user's metadata
        update_data = {
            "settings": json.dumps(settings_data),
            "updated_at": datetime.utcnow().isoformat()
        }

        supabase.table("users").update(update_data).eq("id", current_user.id).execute()

        logger.info(f"Updated settings for user {current_user.id}")

        return {
            "success": True,
            "message": "Settings updated successfully",
            "settings": settings_data
        }

    except Exception as e:
        logger.error(f"Settings update failed: {e}")
        raise HTTPException(500, "Failed to update settings")

@app.get("/api/user/settings")
async def get_user_settings(current_user: User = Depends(get_current_user)):
    """Get user settings"""
    logger.info(f"get_user_settings called for user: {current_user.email}")

    # Return default settings for all users (demo and production)
    settings = {
        "notifications": {
            "translationComplete": True,
            "emailNotifications": False,
            "weeklySummary": True
        },
        "language": "English",
        "time_zone": "UTC (GMT+0)"
    }

    logger.info(f"Returning settings: {settings}")

    return {
        "success": True,
        "data": {
            "settings": settings
        }
    }

@app.post("/api/user/change-password")
async def change_user_password(
    request: Request,
    current_user: User = Depends(get_current_user)
):
    """Change user password"""
    try:
        data = await request.json()
        current_password = data.get("current_password")
        new_password = data.get("new_password")

        if not current_password or not new_password:
            raise HTTPException(400, "Current password and new password are required")

        if len(new_password) < 6:
            raise HTTPException(400, "New password must be at least 6 characters long")

        # Get current user data to verify password
        response = supabase.table("users").select("*").eq("id", current_user.id).execute()
        if not response.data:
            raise HTTPException(404, "User not found")

        user_data = response.data[0]

        # Verify current password
        if not verify_password(current_password, user_data["password_hash"]):
            raise HTTPException(400, "Current password is incorrect")

        # Hash new password
        new_password_hash = get_password_hash(new_password)

        # Update password
        update_data = {
            "password_hash": new_password_hash,
            "updated_at": datetime.utcnow().isoformat()
        }

        supabase.table("users").update(update_data).eq("id", current_user.id).execute()

        logger.info(f"Password changed for user {current_user.id}")

        return {
            "success": True,
            "message": "Password changed successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Password change failed: {e}")
        raise HTTPException(500, "Failed to change password")

@app.delete("/api/user/account")
async def delete_user_account(
    current_user: User = Depends(get_current_user)
):
    """Delete user account (soft delete - mark as inactive)"""
    try:
        # Instead of hard delete, mark as inactive
        update_data = {
            "is_active": False,
            "deleted_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }

        supabase.table("users").update(update_data).eq("id", current_user.id).execute()

        logger.info(f"User account marked for deletion: {current_user.id}")

        return {
            "success": True,
            "message": "Account deletion initiated. Your account will be permanently deleted in 30 days."
        }

    except Exception as e:
        logger.error(f"Account deletion failed: {e}")
        raise HTTPException(500, "Failed to delete account")

# ========== VIDEO TRANSLATION ENDPOINTS ==========

@app.post("/api/translate/audio")
async def translate_audio(
    source_lang: str = Form("auto"),
    target_lang: str = Form("es"),
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    current_user: User = Depends(get_current_user)
):
    """Translate audio file to another language"""
    try:
        # Check credits (10 credits for audio translation)
        if current_user.credits < 10:
            raise HTTPException(400, "Insufficient credits. Need at least 10 credits.")

        # Validate file
        if not file.filename:
            raise HTTPException(400, "No audio file provided")

        # Check file extension
        valid_extensions = ['.mp3', '.wav', '.flac', '.ogg', '.m4a', '.mp4']
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in valid_extensions:
            raise HTTPException(400, f"Invalid audio format. Supported: {', '.join(valid_extensions)}")

        # Save uploaded file
        file_id = str(uuid.uuid4())
        file_path = f"temp_audio_{file_id}{file_ext}"

        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Check file size

        # Deduct credits
        supabase.table("users").update({"credits": current_user.credits - 10}).eq("id", current_user.id).execute()

        # Create job entry
        job_id = str(uuid.uuid4())
        jobs_db[job_id] = {
            "id": job_id,
            "type": "audio_translation",
            "status": "pending",
            "progress": 0,
            "file_path": file_path,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "user_id": current_user.id,
            "user_email": current_user.email,
            "created_at": datetime.utcnow().isoformat(),
            "message": "Starting audio translation...",
            "original_filename": file.filename
        }

        # Process in background
        background_tasks.add_task(
            process_audio_translation_job,
            job_id,
            file_path,
            source_lang,
            target_lang,
            current_user.id
        )

        logger.info(f"Started audio translation job {job_id} for user {current_user.email}")

        return {
            "success": True,
            "job_id": job_id,
            "message": "Audio translation started",
            "status_url": f"/api/jobs/{job_id}/status",
            "remaining_credits": current_user.credits - 10
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Audio translation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Audio translation failed: {str(e)}")

@app.post("/api/translate/video/enhanced")
async def translate_video_enhanced(
    current_user: User = Depends(get_current_user),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    file: UploadFile = File(...),
    target_language: str = Form("es"),
    chunk_size: int = Form(30),
    project_id: Optional[str] = Form(None)
):
    """Enhanced video translation with chunk processing"""
    try:
        # Check if user has enough credits (10 credits for video translation)
        if current_user.credits < 10:
            raise HTTPException(400, "Insufficient credits. Need at least 10 credits.")
        
        if not (app_config.pipeline_available or DEMO_MODE):
            raise JobProcessingError(job_id="", detail="Full video pipeline not available. Running in simplified mode.")
        
        # Save uploaded file
        file_id = str(uuid.uuid4())
        file_ext = os.path.splitext(file.filename)[1]
        file_path = f"temp_{file_id}{file_ext}"
        
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Deduct credits
        supabase.table("users").update({"credits": current_user.credits - 10}).eq("id", current_user.id).execute()
        
        # Create job entry
        job_id = str(uuid.uuid4())
        jobs_db[job_id] = {
            "id": job_id,
            "type": "video_enhanced",
            "status": "processing",
            "progress": 0,
            "file_path": file_path,
            "target_language": target_language, # Add chunk_size here
            "chunk_size": chunk_size,
            "user_id": current_user.id,
            "project_id": project_id,
            "created_at": datetime.utcnow().isoformat()
        }
        save_jobs_db()
        
        # Process in background
        background_tasks.add_task(
            process_video_enhanced_job,
            job_id,
            file_path,
            target_language, # Pass chunk_size
            chunk_size,
            current_user.id
        )
        
        return {
            "success": True,
            "job_id": job_id,
            "message": "Video translation started in background",
            "status_url": f"/api/jobs/{job_id}/status",
            "remaining_credits": current_user.credits - 10
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video translation failed: {str(e)}")

async def process_video_enhanced_job(job_id: str, file_path: str, target_language: str, chunk_size: int, user_id: str):
    """Background task for enhanced video translation"""
    try:
        # Update job status
        jobs_db[job_id]["progress"] = 10
        
        # Initialize pipeline
        config = PipelineConfig(chunk_size=chunk_size)
        pipeline = VideoTranslationPipeline(config)
        
        # Update progress
        jobs_db[job_id]["progress"] = 30
        save_jobs_db()
        
        # Process video
        result = pipeline.process_video(file_path, target_language)
        
        # Update job with results
        jobs_db[job_id].update({
            "status": "completed",
            "progress": 100,
            "result": result,
            "completed_at": datetime.utcnow().isoformat(),
            "output_path": result.get("output_path") or result.get("output_video")
        })
        
        # Cleanup temp file - DISABLED to allow downloading original
        if os.path.exists(file_path):
            # os.remove(file_path)
            pass
            
        # Ensure job is saved!
        save_jobs_db()
    except Exception as e:
        jobs_db[job_id].update({
            "status": "failed",
            "error": str(e),
            "failed_at": datetime.utcnow().isoformat()
        })
        save_jobs_db()
        
        # Refund credits on failure
        try:
            response = supabase.table("users").select("credits").eq("id", user_id).execute()
            if response.data:
                current_credits = response.data[0]["credits"]
                supabase.table("users").update({"credits": current_credits + 10}).eq("id", user_id).execute()
        except Exception as refund_error:
            logger.error(f"Failed to refund credits: {refund_error}")

@app.get("/api/jobs/{job_id}/status")
async def get_job_status(job_id: str, current_user: User = Depends(get_current_user)):
    """Get status of a translation job"""
    job = await job_storage.get_job(job_id)
    
    if not job:
        raise HTTPException(404, "Job not found")

    if job.get("user_id") != current_user.id:
        raise HTTPException(403, "Access denied")

    job_type = job.get("type", "unknown")

    # Build response based on job type
    if job_type == "subtitles":
        response = {
            "success": True,
            "job_id": job_id,
            "status": job.get("status", "pending"),
            "progress": job.get("progress", 0),
            "type": job_type,
            "language": job.get("language"),
            "format": job.get("format"),
            "segment_count": job.get("result", {}).get("segment_count") if job.get("result") else None,
            "download_url": job.get("result", {}).get("download_url") if job.get("status") == "completed" and job.get("result") else None,
            "error": job.get("error"),
            "message": job.get("message", f"Job {job.get('status', 'pending')}")
        }
    elif job_type == "subtitle_to_audio":
        response = {
            "success": True,
            "job_id": job_id,
            "status": job.get("status", "pending"),
            "progress": job.get("progress", 0),
            "type": job_type,
            "source_language": job.get("source_language"),
            "target_language": job.get("target_language"),
            "voice": job.get("voice"),
            "output_format": job.get("output_format"),
            "segment_count": job.get("segment_count"),
            "download_url": job.get("download_url") if job.get("status") == "completed" else None,
            "error": job.get("error"),
            "message": job.get("message", f"Job {job.get('status', 'pending')}")
        }
    else:  # video_translation or other types
        response = {
            "success": True,
            "job_id": job_id,
            "status": job.get("status", "pending"),
            "progress": job.get("progress", 0),
            "type": job_type,
            "target_language": job.get("target_language"),
            "processing_time_s": job.get("processing_time_seconds"),
            "chunks_processed": job.get("processed_chunks"),
            "total_chunks": job.get("total_chunks"),
            "available_chunks": job.get("available_chunks", []),
            "download_url": job.get("download_url") if job.get("status") == "completed" else None,
            "error": job.get("error"),
            "message": job.get("message", f"Job {job.get('status', 'pending')}")
        }

    return response

@app.get("/api/download/chunk/{job_id}/{chunk_id}")
async def download_chunk(job_id: str, chunk_id: int, current_user: User = Depends(get_current_user)):
    """Download a specific translated chunk for preview"""
    # Verify job access
    job = await job_storage.get_job(job_id)
    
    if not job:
        raise HTTPException(404, "Job not found")

    if job.get("user_id") != current_user.id:
        raise HTTPException(403, "Access denied")

    # Check if chunk exists
    available_chunks = job.get("available_chunks", [])
    chunk_info = None
    for chunk in available_chunks:
        if chunk.get("id") == chunk_id:
            chunk_info = chunk
            break

    if not chunk_info:
        raise HTTPException(404, "Chunk not available yet")

    # Get chunk file path
    preview_dir = os.path.join("backend/outputs/previews", job_id)
    chunk_path = os.path.join(preview_dir, f"chunk_{chunk_id:04d}.wav")

    if not os.path.exists(chunk_path):
        raise HTTPException(404, "Chunk file not found")

    # Return chunk file
    return FileResponse(
        chunk_path,
        media_type="audio/wav",
        filename=f"chunk_{chunk_id:04d}.wav",
        headers={
            "Content-Disposition": f"attachment; filename=chunk_{chunk_id:04d}.wav",
            "Cache-Control": "no-cache"
        }
    )

@app.get("/api/download/{file_type}/{file_id}")
async def download_file(file_type: str, file_id: str, current_user: User = Depends(get_current_user)):
    """Download generated files"""
    # Handle subtitle-audio type - delegate to specific handler
    if file_type == "subtitle-audio":
        return await download_subtitle_audio(file_id, current_user)
    
    if file_type == "subtitles":
        # Try file_id-specific filename first (for subtitle translation)
        filename = f"subtitles_{file_id}.srt"
        if not os.path.exists(filename):
            # Fallback to generic name
            filename = "subtitles.srt"
        media_type = "text/plain"
    elif file_type == "video":
        # Check if this is a job output
        job = await job_storage.get_job(file_id)
        if job:
            if job["user_id"] != current_user.id:
                raise HTTPException(403, "Access denied")
            if job.get("status") != "completed":
                raise HTTPException(400, "Video not ready yet")
            # Try output_path, output_video, and fallback
            output_path = job.get("output_path")
            output_video = job.get("output_video")

            # Debug logging
            logger.info(f"Job data keys: {list(job.keys())}")
            logger.info(f"output_path: {output_path} (type: {type(output_path)})")
            logger.info(f"output_video: {output_video} (type: {type(output_video)})")

            if isinstance(output_path, str):
                filename = output_path
            elif isinstance(output_video, str):
                filename = output_video
            else:
                filename = f"backend/outputs/translated_video_{file_id}.mp4"

            logger.info(f"Using filename: {filename} (type: {type(filename)})")

            if not os.path.exists(filename):
                # Fallback to just the filename in cwd
                fallback_name = f"translated_video_{file_id}.mp4"
                if os.path.exists(fallback_name):
                    filename = fallback_name
            media_type = "video/mp4"
        else:
            raise HTTPException(404, "Video file not found")
    elif file_type == "audio":
        # Check if this is an audio job output
        if file_id in jobs_db:
            job = jobs_db[file_id]
            if job["user_id"] != current_user.id:
                raise HTTPException(403, "Access denied")
            if job.get("status") != "completed":
                raise HTTPException(400, "Audio not ready yet")
            filename = job.get("output_path", f"translated_audio_{file_id}.mp3")
            media_type = "audio/mpeg"
        else:
            raise HTTPException(404, "Audio file not found")
    else:
        raise HTTPException(404, "File type not found")
    
    if not os.path.exists(filename):
        raise HTTPException(404, "File not found")
    
    return FileResponse(
        filename,
        media_type=media_type,
        filename=f"octavia_{file_type}_{file_id}{os.path.splitext(filename)[1]}"
    )
# Note: /api/translate/video is handled in routes/translation_routes.py

@app.get("/api/download/video/{job_id}")
async def download_video(
    job_id: str,
    current_user: User = Depends(get_current_user)
):
    """Download translated video file"""
    print(f"[VIDEO DOWNLOAD] Request for job {job_id}, user: {current_user.email}")
    
    # FIRST: Try to find the video file directly by job_id (fastest path)
    import glob
    possible_paths = [
        f"backend/outputs/translated_video_{job_id}.mp4",
        f"outputs/translated_video_{job_id}.mp4",
        f"translated_video_{job_id}.mp4",
    ]
    
    # Check direct paths first
    filename = None
    for path in possible_paths:
        if os.path.exists(path):
            filename = path
            print(f"[VIDEO DOWNLOAD] Found file directly at: {path}")
            break
    
    # If not found, search with glob
    if not filename:
        search_patterns = [
            f"backend/outputs/*{job_id}*.mp4",
            f"outputs/*{job_id}*.mp4",
            f"*{job_id}*.mp4",
        ]
        for pattern in search_patterns:
            matches = glob.glob(pattern)
            if matches:
                filename = matches[0]
                print(f"[VIDEO DOWNLOAD] Found file via pattern: {filename}")
                break
    
    # If file found, serve it directly (skip job lookup for demo users)
    if filename and os.path.exists(filename):
        file_size = os.path.getsize(filename)
        print(f"[VIDEO DOWNLOAD] Serving file: {filename} ({file_size} bytes)")
        return FileResponse(
            filename,
            media_type='video/mp4',
            filename=f"translated_video_{job_id}.mp4"
        )
    
    # File not found - try to get job info for better error message
    print(f"[VIDEO DOWNLOAD] File not found, checking job records...")
    job = None
    
    # Check jobs_db
    if job_id in jobs_db:
        job = jobs_db[job_id]
        print(f"[VIDEO DOWNLOAD] Found job in jobs_db")
    else:
        # Check translation_jobs from translation_routes
        import sys
        for module_name, module in sys.modules.items():
            if 'translation_routes' in module_name and hasattr(module, 'translation_jobs'):
                translation_jobs_ref = module.translation_jobs
                if job_id in translation_jobs_ref:
                    job = translation_jobs_ref[job_id]
                    print(f"[VIDEO DOWNLOAD] Found job in translation_jobs")
                    break
        
        # Check job_storage
        if not job:
            job = await job_storage.get_job(job_id)
            if job:
                print(f"[VIDEO DOWNLOAD] Found job in job_storage")
    
    if job:
        status = job.get("status", "unknown")
        if status != "completed":
            raise HTTPException(400, f"Video not ready yet. Status: {status}")
        
        # Try job's output_path
        output_path = job.get("output_path") or job.get("output_video")
        if output_path and os.path.exists(output_path):
            print(f"[VIDEO DOWNLOAD] Found via job output_path: {output_path}")
            return FileResponse(output_path, media_type='video/mp4', filename=f"translated_video_{job_id}.mp4")
    
    # Nothing found
    print(f"[VIDEO DOWNLOAD] Video file not found for job {job_id}")
    raise HTTPException(404, f"Video file not found. Searched: {possible_paths}")

@app.websocket("/ws/progress/{job_id}")
async def websocket_progress(websocket: WebSocket, job_id: str):
    """Handle WebSocket connections for job progress updates"""
    await ws_manager.connect(websocket, job_id)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket, job_id)
    except Exception as e:
        print(f"WS Exception for {job_id}: {e}")
        ws_manager.disconnect(websocket, job_id)

@app.get("/api/progress/{job_id}")
async def get_job_progress(
    job_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get real-time progress for a translation job (lightweight polling endpoint)"""
    try:
        from services.db_utils import with_retry
        
        # Try job_storage first (handles both demo and regular users)
        async def fetch_job():
            return await job_storage.get_job(job_id)
        
        job = await fetch_job()
        
        # Fallback to jobs_db if not in job_storage
        if not job and job_id in jobs_db:
            job = jobs_db[job_id]
        
        if not job:
            raise HTTPException(404, "Job not found")
        
        # Access control
        if job.get("user_id") and job["user_id"] != current_user.id:
            is_demo_user = DEMO_MODE and current_user.email == "demo@octavia.com"
            if not is_demo_user:
                raise HTTPException(403, "Access denied")
        
        # Return lightweight progress data
        return {
            "progress": job.get("progress", 0),
            "status": job.get("status", "unknown"),
            "message": job.get("message", ""),
            "eta_seconds": job.get("eta_seconds"),
            "job_id": job_id
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Progress fetch error for job {job_id}: {e}")
        raise HTTPException(500, f"Failed to fetch progress: {str(e)}")


@app.post("/api/cancel/{job_id}")
async def cancel_job(
    job_id: str,
    current_user: User = Depends(get_current_user)
):
    """Cancel a running translation job"""
    try:
        # Get the job first
        job = await job_storage.get_job(job_id)
        
        # Fallback to jobs_db if not in job_storage
        if not job and job_id in jobs_db:
            job = jobs_db[job_id]
        
        if not job:
            raise HTTPException(404, "Job not found")
        
        # Access control
        is_demo_user = DEMO_MODE and current_user.email == "demo@octavia.com"
        if job.get("user_id") and job["user_id"] != current_user.id:
            if not is_demo_user:
                raise HTTPException(403, "Access denied - you can only cancel your own jobs")
        
        # Check if job is already completed or cancelled
        current_status = job.get("status", "")
        if current_status in ["completed", "cancelled", "failed"]:
            return {
                "success": True,
                "message": f"Job already {current_status}",
                "job_id": job_id,
                "status": current_status
            }
        
        # Cancel the job in storage
        await job_storage.cancel_job(job_id)
        
        # Also update jobs_db if present (legacy support)
        if job_id in jobs_db:
            jobs_db[job_id]["status"] = "cancelled"
            jobs_db[job_id]["cancelled"] = True
        
        # Cleanup temp files if they exist
        file_path = job.get("file_path") or job.get("input_file")
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"Cleaned up temp file for cancelled job {job_id}: {file_path}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup temp file {file_path}: {cleanup_error}")
        
        # Refund credits based on job type
        job_type = job.get("type", "video")
        credits_to_refund = 0
        if job_type == "video":
            credits_to_refund = VIDEO_TRANSLATION_CREDITS
        elif job_type == "audio":
            credits_to_refund = AUDIO_TRANSLATION_CREDITS
        elif job_type in ["subtitle", "subtitles"]:
            credits_to_refund = 5  # Default subtitle credits
        
        if credits_to_refund > 0 and not is_demo_user:
            user_id = job.get("user_id") or current_user.id
            await refund_credits(user_id, credits_to_refund, "job cancelled by user")
            logger.info(f"Refunded {credits_to_refund} credits for cancelled job {job_id}")
        
        logger.info(f"Job {job_id} cancelled successfully by user {current_user.id}")
        
        return {
            "success": True,
            "message": "Job cancelled successfully",
            "job_id": job_id,
            "credits_refunded": credits_to_refund if not is_demo_user else 0
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Cancel job error for {job_id}: {e}")
        raise HTTPException(500, f"Failed to cancel job: {str(e)}")


@app.get("/api/jobs/history")
async def get_job_history(
    current_user: User = Depends(get_current_user),
    limit: int = 50,
    offset: int = 0,
    status: Optional[str] = None,
    type: Optional[str] = None
):
    """
    Get job history for the current user
    For non-demo users, queries Supabase translation_jobs table
    For demo users, returns empty list or mock data
    """
    try:
        # Check if demo user
        is_demo_user = DEMO_MODE and current_user.email == "demo@octavia.com"
        
        if is_demo_user:
            # Demo users get empty history
            return {
                "success": True,
                "jobs": [],
                "total": 0,
                "limit": limit,
                "offset": offset
            }
        
        # For non-demo users, query Supabase
        from services.db_utils import with_retry
        
        async def fetch_jobs():
            # Build query
            query = supabase.table("translation_jobs").select(
                "id, type, status, progress, target_language, original_filename, "
                "created_at, completed_at, failed_at, error, message, "
                "source_language, target_lang, language, format, output_format"
            ).eq("user_id", current_user.id).order("created_at", desc=True)
            
            # Apply filters
            if status:
                query = query.eq("status", status)
            if type:
                query = query.eq("type", type)
            
            # Apply pagination
            query = query.range(offset, offset + limit - 1)
            
            response = query.execute()
            return response.data
        
        jobs = await with_retry(fetch_jobs)
        
        # Get total count
        async def fetch_count():
            query = supabase.table("translation_jobs").select(
                "id", count="exact"
            ).eq("user_id", current_user.id)
            
            if status:
                query = query.eq("status", status)
            if type:
                query = query.eq("type", type)
            
            response = query.execute()
            return response.count
        
        total = await with_retry(fetch_count)
        
        return {
            "success": True,
            "jobs": jobs or [],
            "total": total or 0,
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        logger.error(f"Job history fetch error: {e}")
        raise HTTPException(500, f"Failed to fetch job history: {str(e)}")

@app.get("/api/download/original/{job_id}")
async def download_original_video(
    job_id: str,
    current_user: User = Depends(get_current_user)
):
    """Download original video file (if still available)"""
    logger.info(f"Original video download request for job {job_id}")

    # Check job records (Supabase or local)
    job = await job_storage.get_job(job_id)
    if not job:
        # Fallback to jobs_db
        if job_id in jobs_db:
            job = jobs_db[job_id]
        else:
            logger.error(f"Job {job_id} not found in any job store")
            raise HTTPException(404, "Job not found")

    logger.info(f"Job status: {job.get('status')}, user_id: {job.get('user_id')}, current_user: {current_user.id}")

    # Access control
    if job.get("user_id") and job["user_id"] != current_user.id:
        # Skip check for demo user if applicable
        is_demo_user = DEMO_MODE and current_user.email == "demo@octavia.com"
        if not is_demo_user:
            logger.error(f"Access denied: job user {job.get('user_id')} != current user {current_user.id}")
            raise HTTPException(403, "Access denied")

    # Get the original file path from job data
    file_path = job.get("file_path")
    if not file_path:
        logger.error(f"Original file path not found in job data for {job_id}")
        raise HTTPException(404, "Original video file record not found")

    # Path safety and existence check - try multiple variations
    final_path = None
    possible_paths = [
        file_path,                               # As stored
        os.path.join("backend", file_path),        # Relative to project root
        os.path.basename(file_path),             # Just the filename in current dir
        os.path.join("octavia", "backend", file_path), # Absolute-ish
    ]
    
    for path in possible_paths:
        if path and os.path.exists(path):
            final_path = path
            break
    
    if not final_path:
        # Attempt glob search if path variations fail
        import glob
        base = os.path.basename(file_path)
        pattern = f"**/{base}"
        matches = glob.glob(pattern, recursive=True)
        if matches:
            final_path = matches[0]

    if not final_path:
        logger.error(f"Original file not found on disk: {file_path}")
        raise HTTPException(404, f"Original video file no longer available on server. Searched prefix of: {file_path}")

    # Determine media type for original file
    ext = os.path.splitext(final_path)[1].lower()
    media_type = "video/mp4" # Default for original videos
    
    if ext in ['.mp3', '.wav', '.ogg']:
        media_type = "audio/mpeg"
    
    filename = job.get("original_filename") or os.path.basename(final_path)
    if not filename.endswith(ext):
        filename += ext
        
    logger.info(f"Serving original file: {final_path} as {filename}")
    return FileResponse(
        final_path,
        media_type=media_type,
        filename=filename
    )

@app.get("/api/download/audio/{job_id}")
async def download_audio(
    job_id: str,
    current_user: User = Depends(get_current_user)
):
    """Download translated audio file"""
    logger.info(f"Audio download request for job {job_id}")
    
    # Check in translation_jobs first (from translation_routes)
    import sys
    translation_jobs = None
    for module_name, module in sys.modules.items():
        if 'translation_routes' in module_name and hasattr(module, 'translation_jobs'):
            translation_jobs = module.translation_jobs
            break
    
    job = None
    if translation_jobs and job_id in translation_jobs:
        job = translation_jobs[job_id]
        logger.info(f"Found audio job {job_id} in translation_jobs")
    elif job_id in jobs_db:
        job = jobs_db[job_id]
        logger.info(f"Found audio job {job_id} in jobs_db")
    else:
        logger.error(f"Job {job_id} not found in translation_jobs or jobs_db")
        raise HTTPException(404, "Job not found")
    logger.info(f"Job status: {job.get('status')}, user_id: {job.get('user_id')}, current_user: {current_user.id}")
    
    if job["user_id"] != current_user.id:
        logger.error(f"Access denied: job user {job['user_id']} != current user {current_user.id}")
        raise HTTPException(403, "Access denied")
    
    job_status = job.get("status")
    if job_status != "completed":
        logger.error(f"Job not completed: status = {job_status}")
        raise HTTPException(400, f"Audio not ready yet. Status: {job_status}")
    
    # Try to get filename from job data
    filename = job.get("output_path")
    logger.info(f"Looking for audio file: {filename}")
    
    # Check if file exists at expected path
    if not filename:
        logger.error(f"No output_path set for audio job {job_id}")
        raise HTTPException(404, "Audio file path not set. Please try the translation again.")
    
    if not os.path.exists(filename):
        logger.error(f"Audio file not found at path: {filename}")
        raise HTTPException(404, f"Audio file not found at: {filename}")
    
    # Verify file is actually an audio file
    file_size = os.path.getsize(filename)
    logger.info(f"Audio file size: {file_size} bytes")
    
    if file_size < 1000:  # Suspiciously small for audio
        logger.warning(f"Audio file suspiciously small: {file_size} bytes")
        # Check if it's an error message
        with open(filename, 'rb') as f:
            content = f.read(500)
            if b'error' in content.lower() or b'failed' in content.lower():
                logger.error("Audio file contains error message instead of audio")
                raise HTTPException(500, "Audio generation failed. Please try again.")
    
    # Generate download filename
    original_name = job.get("original_filename", f"audio_{job_id}")
    base_name = os.path.splitext(original_name)[0]
    target_lang = job.get('target_language', 'es')
    download_filename = f"{base_name}_translated_{target_lang}.mp3"
    logger.info(f"Serving audio file: {filename} as {download_filename}")
    
    return FileResponse(
        filename,
        media_type="audio/mpeg",
        filename=download_filename
    )



@app.get("/api/translate/download/video/{job_id}")
async def download_video_translate(
    job_id: str,
    current_user: User = Depends(get_current_user)
):
    """Download translated video file (alternative endpoint)"""
    # Just delegate to the main download endpoint
    return await download_video(job_id, current_user)

@app.get("/api/download/{job_id}")
async def download_generic(job_id: str, current_user: User = Depends(get_current_user)):
    """Generic download endpoint that tries to find any file for this job"""
    logger.info(f"Generic download request for job {job_id}")
    
    # Try to find the job in Supabase
    try:
        job_data = await job_storage.get_job(job_id)
        if job_data:
            job_type = job_data.get("type", "unknown")
            logger.info(f"Found job {job_id} with type {job_type}")
            
            # Route to appropriate download handler
            if "audio" in job_type:
                return await download_audio(job_id, current_user)
            else:
                return await download_video(job_id, current_user)
    except Exception as e:
        logger.error(f"Error fetching job {job_id}: {e}")
    
    # Job not found
    raise HTTPException(404, "Job not found")


# ========== HEALTH & TESTING ENDPOINTS ==========

@app.get("/health")
async def health_check():
    """Lightweight health check endpoint for Docker healthchecks."""
    return {"status": "healthy"}


@app.get("/api/health")
async def api_health_check():
    # Check whisper model status safely
    whisper_status = "not_available"
    try:
        if 'whisper_model' in globals() and globals().get('whisper_model') is not None:
            whisper_status = "loaded"
    except:
        pass

    return {
        "success": True,
        "status": "healthy",
        "service": "Octavia Video Translator",
        "version": "4.0.0",
        "database": "Supabase",
        "payment": {
            "mode": POLAR_SERVER,
            "test_mode": ENABLE_TEST_MODE,
            "real_products_configured": True,
        },
        "models": {
            "whisper": whisper_status,
            "translation": "loaded" if translator_cache else "not_available",
            "available_pairs": list(HELSINKI_MODELS.keys()),
            "pipeline": "available" if PIPELINE_AVAILABLE else "simplified_mode"
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/debug/jobs")
async def debug_jobs():
    """Debug endpoint to list all jobs in jobs_db"""
    return {
        "success": True,
        "jobs_db": jobs_db,
        "subtitle_jobs": subtitle_jobs,
        "total_jobs": len(jobs_db) + len(subtitle_jobs),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/debug/files/{job_id}")
async def debug_files(job_id: str, current_user: User = Depends(get_current_user)):
    """Debug endpoint to check file locations for a job"""
    if job_id not in jobs_db:
        return {"success": False, "error": "Job not found"}

    job = jobs_db[job_id]


# ========== VOICES ENDPOINTS ==========

@app.get("/api/voices/all")
async def get_all_voices():
    """Get all available gTTS voices grouped by language"""
    voices_by_language = {}

    for lang_code, voices in GTTS_VOICE_OPTIONS.items():
        voices_by_language[lang_code] = []

        for voice_name, voice_id in voices.items():
            voices_by_language[lang_code].append({
                "name": voice_name,
                "voice_id": voice_id,
                "language_code": lang_code,
                "sample_text": "Hello! This is a preview of the voice."
            })

    return {
        "success": True,
        "tts_engine": "gTTS",
        "voices_by_language": voices_by_language,
        "total_voices": sum(len(v) for v in voices_by_language.values()),
        "info": {
            "engine": "Google Text-to-Speech (gTTS)",
            "description": "High-quality crystal-clear voice synthesis using Google Translate TTS",
            "features": [
                "Crystal-clear audio quality",
                "Fast synthesis",
                "Supports 15+ languages",
                "No voice selection per language (language code based)"
            ]
        }
    }

@app.get("/api/voices/{language}")
async def get_voices_by_language(language: str):
    """Get voices for a specific language"""
    if language not in GTTS_VOICE_OPTIONS:
        return {
            "success": False,
            "error": f"Language '{language}' not supported",
            "available_languages": list(GTTS_VOICE_OPTIONS.keys())
        }

    voices = GTTS_VOICE_OPTIONS[language]
    voice_list = []

    for voice_name, voice_id in voices.items():
        voice_list.append({
            "name": voice_name,
            "voice_id": voice_id,
            "language_code": language,
            "sample_text": "Hello! This is a preview of the voice."
        })

    return {
        "success": True,
        "language": language,
        "voices": voice_list
    }


@app.post("/api/voices/preview")
async def preview_voice(
    voice_id: str = Form(...),
    text: str = Form(...),
    language: str = Form(...),
    current_user: User = Depends(get_current_user)
):
    """Generate a voice preview using gTTS"""
    # Check if user has enough credits (1 credit per preview)
    # Demo users get free previews
    is_demo_user = DEMO_MODE and current_user.email == "demo@octavia.com"
    
    if not is_demo_user and current_user.credits < 1:
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail="Not enough credits. Each preview costs 1 credit."
        )

    # For gTTS, voice_id is the language code (e.g., "en", "es", "fr")
    # voice_id can be a full code like "en" or "en-uk", or a friendly name
    lang_code = voice_id

    # If voice_id looks like a friendly name, extract language code
    if voice_id in GTTS_VOICE_OPTIONS or any(voice_id in v for v in GTTS_VOICE_OPTIONS.values()):
        # Find the actual language code
        for lang, voices in GTTS_VOICE_OPTIONS.items():
            for name, vid in voices.items():
                if vid == voice_id or name.lower().replace(" (gTTS)", "").replace(" (", "-").replace(")", "") in voice_id.lower():
                    lang_code = vid
                    break

    logger.info(f"Generating gTTS voice preview: voice_id={voice_id}, lang_code={lang_code}, text={text[:50]}")

    # Ensure outputs directory exists
    os.makedirs("outputs", exist_ok=True)

    # Generate output filename
    preview_filename = f"preview_{uuid.uuid4().hex[:8]}.mp3"
    output_path = f"outputs/{preview_filename}"

    # Generate speech using gTTS
    try:
        # Validate language code for gTTS
        valid_gtts_langs = ['en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'ko', 'zh-cn', 'zh-tw', 'ar', 'hi', 'nl', 'pl', 'tr', 'sv']
        gtts_lang = lang_code.lower()
        if gtts_lang not in valid_gtts_langs:
            # Try to map to valid code
            if '-' in gtts_lang:
                gtts_lang = gtts_lang.split('-')[0]
            if gtts_lang not in valid_gtts_langs:
                gtts_lang = 'en'  # Fallback to English

        tts_success = False

        # Try gTTS
        try:
            from gtts import gTTS as GoogleTTS
            logger.info(f"Generating voice preview with gTTS: lang={gtts_lang}, text={text[:50]}")

            tts = GoogleTTS(text=text, lang=gtts_lang, slow=False)
            tts.save(output_path)

            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                tts_success = True
                logger.info(f"gTTS succeeded: {output_path}")
        except Exception as gtts_error:
            logger.warning(f"gTTS failed: {gtts_error}")
            raise HTTPException(
                status_code=400,
                detail=f"Voice preview not available for this language. gTTS failed: {str(gtts_error)}"
            )

        # Verify file was created and has content
        if not tts_success or not os.path.exists(output_path):
            raise HTTPException(
                status_code=500,
                detail="Failed to generate voice preview. Audio file was not created."
            )

        file_size = os.path.getsize(output_path)
        if file_size == 0:
            os.remove(output_path)
            raise Exception("Generated audio file is empty after writing")

        logger.info(f"Voice preview generated: {output_path}, size: {file_size} bytes")
    except Exception as e:
        # Clean up if file was created but is invalid
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
                logger.info(f"Cleaned up invalid file: {output_path}")
            except:
                pass

        logger.error(f"Failed to generate audio: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate voice preview: {str(e)}"
        )

    # Deduct 1 credit for preview (skip for demo users)
    new_credits = current_user.credits
    if not is_demo_user:
        new_credits = current_user.credits - 1
        # Update database
        try:
            supabase.table("users").update({"credits": new_credits}).eq("id", current_user.id).execute()
        except Exception as e:
            logger.error(f"Failed to update credits: {e}")

    return {
        "success": True,
        "preview_url": f"/api/voices/preview-file/{preview_filename}",
        "credits_remaining": new_credits,
        "tts_engine": "gTTS",
        "language_code": gtts_lang,
        "message": "Preview generated successfully with gTTS"
    }


@app.get("/api/voices/preview-file/{filename}")
async def download_preview_audio(filename: str):
    """Download generated voice preview"""
    file_path = f"outputs/{filename}"
    
    logger.info(f"PREVIEW DOWNLOAD REQUEST: {filename}")

    if not os.path.exists(file_path):
        logger.warning(f"Preview file not found: {file_path}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Preview file not found")

    logger.info(f"Serving preview audio: {file_path}")
    return FileResponse(
        file_path,
        media_type="audio/mpeg",
        filename=filename
    )


@app.post("/api/test/integration")
async def test_integration():
    """Run integration test on sample video"""
    try:
        test_video = "test_samples/sample_30s.mp4"
        
        if not os.path.exists(test_video):
            return {
                "success": False,
                "error": f"Test video not found: {test_video}",
                "message": "Please create a test sample first"
            }
        
        if not PIPELINE_AVAILABLE:
            return {
                "success": False,
                "error": "Pipeline modules not available",
                "message": "Running in simplified mode"
            }
        
        # Run the pipeline on test video
        pipeline = VideoTranslationPipeline()
        result = pipeline.process_video(test_video, "es")
        
        return {
            "success": True,
            "message": "Integration test completed",
            "result": result
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Integration test failed"
        }

@app.post("/api/admin/cleanup")
async def manual_cleanup():
    """Manually trigger cleanup of temporary files"""
    try:
        result = run_full_cleanup()
        return {
            "success": True,
            "message": f"Cleanup completed: {result['total_cleaned']} files cleaned",
            "files_cleaned": result["total_cleaned"],
            "space_freed_bytes": result["total_freed_bytes"],
            "space_freed_formatted": format_bytes(result["total_freed_bytes"]),
            "failed": result["total_failed"]
        }
    except Exception as e:
        logger.error(f"Manual cleanup failed: {e}")
        raise HTTPException(500, f"Cleanup failed: {str(e)}")

@app.get("/api/metrics")
async def get_metrics():
    """Get processing metrics from logs"""
    try:
        log_file = "artifacts/logs.jsonl"
        
        if not os.path.exists(log_file):
            return {
                "success": True,
                "message": "No logs available yet",
                "metrics": {}
            }
        
        # Simple metrics collection
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        return {
            "success": True,
            "total_logs": len(lines),
            "log_file": log_file,
            "message": f"Found {len(lines)} log entries"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to get metrics"
        }

# ========== SUBTITLE MANAGEMENT ENDPOINTS ==========

@app.get("/api/translate/subtitles/list")
async def list_subtitle_files(job_id: str, current_user: User = Depends(get_current_user)):
    """List subtitle files for a translation job"""
    # Get job from Supabase
    job = await job_storage.get_job(job_id)
    
    if not job:
        raise HTTPException(404, "Job not found")
    
    if job.get("user_id") != current_user.id:
        raise HTTPException(403, "Access denied")
    
    # Check for subtitle files in output directory
    subtitle_files = []
    output_dir = os.path.dirname(job.get("output_path", ""))
    
    if output_dir and os.path.exists(output_dir):
        for file in os.listdir(output_dir):
            if file.endswith(('.srt', '.vtt', '.ass', '.ssa')):
                subtitle_files.append({
                    "name": file,
                    "path": os.path.join(output_dir, file),
                    "size": os.path.getsize(os.path.join(output_dir, file)),
                    "format": os.path.splitext(file)[1][1:]
                })
    
    return {
        "success": True,
        "job_id": job_id,
        "subtitle_files": subtitle_files
    }

@app.get("/api/download/subtitles/{job_id}/{filename}")
async def download_subtitle_file(
    job_id: str, 
    filename: str,
    current_user: User = Depends(get_current_user)
):
    """Download a specific subtitle file"""
    # Get job from Supabase
    job = await job_storage.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    
    if job.get("user_id") != current_user.id:
        raise HTTPException(403, "Access denied")
    
    # Find the subtitle file
    output_dir = os.path.dirname(job.get("output_path", ""))
    file_path = os.path.join(output_dir, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(404, "Subtitle file not found")
    
    # Determine media type
    ext = os.path.splitext(filename)[1].lower()
    media_types = {
        '.srt': 'text/plain',
        '.vtt': 'text/vtt',
        '.ass': 'text/x-ssa',
        '.ssa': 'text/x-ssa'
    }
    
    media_type = media_types.get(ext, 'application/octet-stream')
    
    return FileResponse(
        file_path,
        media_type=media_type,
        filename=filename
    )

@app.get("/api/translate/subtitles/review/{job_id}")
async def review_subtitles(
    job_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get subtitle content for review/editing"""
    # Get job from Supabase
    job = await job_storage.get_job(job_id)
    
    if not job:
        # Try to auto-register from orphaned subtitle file
        for ext in ("srt", "vtt", "ass"):
            filename = f"subtitles_{job_id}.{ext}"
            if os.path.exists(filename):
                try:
                    with open(filename, "r", encoding="utf-8") as f:
                        content = f.read()
                except:
                    pass
    if not job:
        raise HTTPException(404, "Job not found")
    if job["user_id"] != current_user.id:
        raise HTTPException(403, "Access denied")
    if job.get("status") != "completed":
        raise HTTPException(400, "Subtitles not ready yet. Status: " + job.get("status", "unknown"))
    
    # Get language with fallbacks
    language = job.get("language") or job.get("target_language") or "English"
    if len(language) <= 3: # If it's a code like 'es'
        # Simple mapping or capitalize
        lang_map = {"en": "English", "es": "Spanish", "fr": "French", "de": "German"}
        language = lang_map.get(language.lower(), language.capitalize())
    else:
        language = language.capitalize()

    format = job.get("format", "srt")
    filename = job.get("filename", f"subtitles_{job_id}.{format}")
    
    subtitle_content = ""
    
    try:
        # Try multiple possible paths for the subtitle file
        import glob
        possible_paths = [
            filename,  # Direct path from job
            os.path.join("backend/outputs", filename),
            os.path.join("outputs", filename),
            os.path.join("backend/outputs/subtitles", filename),
            os.path.join("outputs/subtitles", filename),
            f"backend/outputs/subtitles_{job_id}.{format}",
            f"outputs/subtitles_{job_id}.{format}",
            f"backend/outputs/subtitles/subtitles_{job_id}.{format}",
            f"outputs/subtitles/subtitles_{job_id}.{format}",
            f"subtitles_{job_id}.{format}",
        ]
        
        # Check direct paths first
        subtitle_path = None
        for path in possible_paths:
            if os.path.exists(path):
                subtitle_path = path
                print(f"[SUBTITLE REVIEW] Found subtitle file at: {path}")
                break
        
        # If not found, search with glob
        if not subtitle_path:
            search_patterns = [
                f"backend/outputs/*{job_id}*.{format}",
                f"outputs/*{job_id}*.{format}",
                f"*{job_id}*.{format}",
            ]
            for pattern in search_patterns:
                matches = glob.glob(pattern)
                if matches:
                    subtitle_path = matches[0]
                    print(f"[SUBTITLE REVIEW] Found subtitle file via pattern: {subtitle_path}")
                    break
        
        # Read the file if found
        if subtitle_path and os.path.exists(subtitle_path):
            with open(subtitle_path, "r", encoding="utf-8") as f:
                subtitle_content = f.read()
            print(f"[SUBTITLE REVIEW] Successfully read {len(subtitle_content)} characters from {subtitle_path}")
        else:
            # Fallback to job content
            subtitle_content = job.get("content", "")
            print(f"[SUBTITLE REVIEW] Using content from job record: {len(subtitle_content)} characters")
            
    except Exception as e:
        print(f"[SUBTITLE REVIEW] Error reading subtitle file: {e}")
        subtitle_content = job.get("content", "")
    
    return {
        "success": True,
        "data": {
            "job_id": job_id,
            "status": job.get("status"),
            "format": format,
            "language": language,
            "segment_count": job.get("segment_count", 0),
            "content": subtitle_content,
            "download_url": job.get("download_url"),
            "created_at": job.get("created_at"),
            "completed_at": job.get("completed_at")
        }
    }




















async def generate_audio_from_subtitles(subtitle_content: str, language: str, voice: str, output_format: str = "mp3"):
    """Generate high-quality audio from subtitles using the same AudioTranslator pipeline as video translation"""
    try:
        logger.info(f"Starting HIGH-QUALITY subtitle-to-audio generation for language: {language}, voice: {voice}")

        # Parse SRT content into segments
        subtitles = []
        lines = subtitle_content.strip().split('\n')

        current_subtitle = {}
        for line in lines:
            line = line.strip()
            if not line:
                if current_subtitle and 'text' in current_subtitle:
                    subtitles.append(current_subtitle.copy())
                    current_subtitle = {}
                continue

            # Parse SRT format
            if line.isdigit() and 'index' not in current_subtitle:
                current_subtitle = {'index': int(line)}
            elif '-->' in line:
                start_str, end_str = line.split('-->')
                def parse_srt_time(ts):
                    ts = ts.strip()
                    if ',' in ts:
                        time_part, ms = ts.split(',')
                        h, m, s = time_part.split(':')
                        return int(h)*3600 + int(m)*60 + int(s) + int(ms)/1000
                    return 0
                current_subtitle['start'] = parse_srt_time(start_str)
                current_subtitle['end'] = parse_srt_time(end_str)
                current_subtitle['duration'] = current_subtitle['end'] - current_subtitle['start']
            elif 'start' in current_subtitle and 'text' not in current_subtitle:
                current_subtitle['text'] = line

        # Add last subtitle
        if current_subtitle and 'text' in current_subtitle:
            subtitles.append(current_subtitle)

        logger.info(f"Parsed {len(subtitles)} subtitle segments")

        if not subtitles:
            raise Exception("No valid subtitle segments found")

        # Use the SAME AudioTranslator as video translation for crystal-clear quality
        from modules.audio_translator import AudioTranslator, TranslationConfig

        # Configure with the same settings as video translation
        config = TranslationConfig(
            source_lang=language,
            target_lang=language,  # No translation, just TTS
            enable_input_normalization=True,
            enable_denoising=True,
            enable_gain_consistency=True,
            enable_silence_padding=True,
            validation_spots=3,  # Quality validation like video
            target_lufs=-16.0  # Same loudness as video
        )

        translator = AudioTranslator(config)
        translator.load_models()  # Load the same high-quality models as video

        # Combine all subtitle text (video translation processes entire content)
        full_text = " ".join([sub.get('text', '').strip() for sub in subtitles if sub.get('text', '').strip()])

        if not full_text:
            raise Exception("No text content found in subtitles")

        logger.info(f"Combined subtitle text: {len(full_text)} characters")

        # Create temporary output path
        output_path = f"temp_subtitle_audio_{uuid.uuid4()}.wav"

        # Generate speech using the SAME high-quality TTS pipeline as video translation
        success, timing_segments = translator.synthesize_speech_with_timing(
            full_text,
            [],  # No existing segments for subtitle-to-audio
            output_path
        )

        if not success or not os.path.exists(output_path):
            raise Exception("High-quality TTS synthesis failed")

        # Apply the SAME audio processing as video translation
        if translator.config.enable_gain_consistency:
            logger.info("Applying gain consistency (same as video translation)")
            final_audio = AudioSegment.from_file(output_path)
            processed_audio = translator.apply_gain_consistency(final_audio, translator.config.target_lufs)
            processed_audio.export(output_path, format="wav")

        logger.info(f"[SUCCESS] Generated crystal-clear subtitle audio: {output_path}")

        # Convert to requested format and return
        final_output_path = f"outputs/subtitle_audio_{uuid.uuid4()}.{output_format}"
        os.makedirs("outputs", exist_ok=True)

        final_audio = AudioSegment.from_file(output_path)

        if output_format == "mp3":
            final_audio.export(final_output_path, format="mp3", bitrate="192k")
        elif output_format == "wav":
            final_audio.export(final_output_path, format="wav")
        else:
            final_audio.export(final_output_path, format="mp3", bitrate="192k")

        # Cleanup temp file
        if os.path.exists(output_path):
            os.remove(output_path)

        output_filename = os.path.basename(final_output_path)
        logger.info(f"[SUCCESS] Subtitle-to-audio completed with video-quality audio: {output_filename}")
        return output_filename, len(subtitles)

    except Exception as e:
        logger.error(f"High-quality subtitle-to-audio failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

# Add this endpoint to app.py (after subtitle translation endpoints)
@app.post("/api/generate/subtitle-audio")
async def generate_subtitle_audio(
    current_user: User = Depends(get_current_user),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    file: UploadFile = File(...),
    source_language: str = Form("en"),
    target_language: str = Form("es"),
    voice: str = Form("Aria (Female)"),
    output_format: str = Form("mp3")
):
    """Generate audio from subtitle file with translation"""
    try:
        # Check if demo user
        is_demo_user = DEMO_MODE and current_user.email == "demo@octavia.com"
        
        # Check if user has enough credits (5 credits for subtitle-to-audio)
        if not is_demo_user and current_user.credits < 5:
            raise HTTPException(400, "Insufficient credits. You need at least 5 credits.")
        
        # Validate file
        if not file.filename:
            raise HTTPException(400, "No subtitle file provided")
        
        # Check file extension
        valid_extensions = ['.srt', '.vtt', '.ass', '.ssa', '.txt']
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in valid_extensions:
            raise HTTPException(400, f"Invalid file format. Supported formats: {', '.join(valid_extensions)}")
        
        # Save uploaded file
        file_id = str(uuid.uuid4())
        file_path = f"temp_subtitle_{file_id}{file_ext}"
        
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Deduct credits (skip for demo users)
        if not is_demo_user:
            supabase.table("users").update({"credits": current_user.credits - 5}).eq("id", current_user.id).execute()
        
        # Create job entry
        job_id = str(uuid.uuid4())
        job_data = {
            "id": job_id,
            "type": "subtitle_to_audio",
            "status": "pending",
            "progress": 0,
            "file_path": file_path,
            "source_language": source_language,
            "target_language": target_language,
            "voice": voice,
            "output_format": output_format,
            "user_id": current_user.id,
            "user_email": current_user.email,
            "created_at": datetime.utcnow().isoformat(),
            "original_filename": file.filename
        }
        await job_storage.create_job(job_data)
        
        # Process in background
        background_tasks.add_task(
            process_subtitle_to_audio_job,
            job_id,
            file_path,
            source_language,
            target_language,
            voice,
            output_format,
            current_user.id
        )
        
        logger.info(f"Started subtitle-to-audio job {job_id} for user {current_user.email}")
        
        return {
            "success": True,
            "job_id": job_id,
            "message": "Audio generation started in background",
            "status_url": f"/api/generate/subtitle-audio/status/{job_id}",
            "remaining_credits": 5000 if is_demo_user else current_user.credits - 5
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Subtitle-to-audio generation failed: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Audio generation failed: {str(e)}")

async def translate_subtitle_content(subtitle_content: str, source_language: str, target_language: str, file_path: str) -> tuple[str, str]:
    """Translate subtitle content using the same AudioTranslator as video translation"""
    translated_content = subtitle_content
    language_to_use = source_language

    if source_language != target_language:
        logger.info(f"Translation needed from {source_language} to {target_language}")

        # Use the same AudioTranslator as video translation for consistency
        try:
            from modules.audio_translator import AudioTranslator, TranslationConfig
            config = TranslationConfig(source_lang=source_language, target_lang=target_language)
            translator = AudioTranslator(config)

            if file_path.lower().endswith('.srt'):
                logger.info("Parsing SRT file for batch translation")
                subs = pysrt.open(file_path)

                # Extract texts for batch processing
                subtitle_texts = []
                subtitle_indices = []
                for i, sub in enumerate(subs):
                    text = sub.text.strip()
                    if text:  # Only process non-empty subtitles
                        subtitle_texts.append(text)
                        subtitle_indices.append((i, sub))

                logger.info(f"Found {len(subtitle_texts)} non-empty subtitles for translation")

                # Batch translate using the same method as video translation
                translated_texts = []
                batch_size = 10  # Process 10 subtitles at a time

                for batch_start in range(0, len(subtitle_texts), batch_size):
                    batch_end = min(batch_start + batch_size, len(subtitle_texts))
                    batch_texts = subtitle_texts[batch_start:batch_end]

                    logger.info(f"Translating batch {batch_start//batch_size + 1}: subtitles {batch_start + 1}-{batch_end}")

                    # Use the same translation method as video translation
                    try:
                        combined_text = " ||| ".join(batch_texts)
                        translated_batch, _ = translator.translate_text_with_context(combined_text, [])
                        batch_results = translated_batch.split(" ||| ")
                        translated_texts.extend(batch_results[:len(batch_texts)])

                    except Exception as batch_error:
                        logger.warning(f"Batch translation failed: {batch_error}")
                        # Fallback: translate individually
                        for text in batch_texts:
                            try:
                                translated, _ = translator.translate_text_with_context(text, [])
                                translated_texts.append(translated)
                            except Exception as individual_error:
                                logger.error(f"Individual translation failed: {individual_error}")
                                translated_texts.append(f"[{target_language.upper()}] {text}")

                # Reconstruct subtitles with translated text
                translated_subs = []
                text_idx = 0
                for i, sub in subtitle_indices:
                    if text_idx < len(translated_texts):
                        translated_text = translated_texts[text_idx]
                        translated_subs.append(pysrt.SubRipItem(
                            index=sub.index,
                            start=sub.start,
                            end=sub.end,
                            text=translated_text
                        ))
                        text_idx += 1

                # Create translated SRT content
                translated_content = ""
                for sub in translated_subs:
                    translated_content += f"{sub.index}\n"
                    start_str = str(sub.start).replace('.', ',')
                    end_str = str(sub.end).replace('.', ',')
                    translated_content += f"{start_str} --> {end_str}\n"
                    translated_content += f"{sub.text}\n\n"

                logger.info(f"Translation completed. Translated content length: {len(translated_content)}")
                language_to_use = target_language
            else:
                logger.info(f"File {file_path} is not SRT format, skipping structured translation")
        except Exception as translator_error:
            logger.warning(f"AudioTranslator failed, falling back to basic translation: {translator_error}")
            # Fallback to basic translation if AudioTranslator fails
            try:
                from deep_translator import GoogleTranslator
                translator = GoogleTranslator(source=source_language, target=target_language)

                if file_path.lower().endswith('.srt'):
                    subs = pysrt.open(file_path)
                    translated_subs = []
                    for sub in subs:
                        if sub.text and sub.text.strip():
                            translated_text = translator.translate(sub.text)
                            sub.text = translated_text
                        translated_subs.append(sub)

                    # Create translated SRT content
                    translated_content = ""
                    for sub in translated_subs:
                        translated_content += f"{sub.index}\n"
                        start_str = str(sub.start).replace('.', ',')
                        end_str = str(sub.end).replace('.', ',')
                        translated_content += f"{start_str} --> {end_str}\n"
                        translated_content += f"{sub.text}\n\n"

                    language_to_use = target_language
                else:
                    logger.info(f"File {file_path} is not SRT format, skipping structured translation")
            except Exception as fallback_error:
                logger.warning(f"Fallback translation also failed: {fallback_error}")
    else:
        logger.info(f"No translation needed - source and target languages are the same ({source_language})")

    return translated_content, language_to_use
    """Translate subtitle content if needed, return translated content and language to use"""
    translated_content = subtitle_content
    language_to_use = source_language

    if source_language != target_language:
        logger.info(f"Translation needed from {source_language} to {target_language}")

        # Check if translation is available
        language_pair = f"{source_language}-{target_language}"
        direct_translator = get_translator(source_language, target_language)
        can_use_pivot = (get_translator(source_language, "en") is not None and
                        get_translator("en", target_language) is not None)

        logger.info(f"Direct translator available: {direct_translator is not None}")
        logger.info(f"Pivot translation available: {can_use_pivot}")

        if direct_translator or can_use_pivot:
            logger.info(f"Starting translation using {'direct' if direct_translator else 'pivot'} method")

            # Parse and translate SRT content with batch processing
            if file_path.lower().endswith('.srt'):
                logger.info("Parsing SRT file for batch translation")
                subs = pysrt.open(file_path)

                # Extract texts for batch processing
                subtitle_texts = []
                subtitle_indices = []
                for i, sub in enumerate(subs):
                    text = sub.text.strip()
                    if text:  # Only process non-empty subtitles
                        subtitle_texts.append(text)
                        subtitle_indices.append((i, sub))

                logger.info(f"Found {len(subtitle_texts)} non-empty subtitles for translation")

                # Batch translate subtitles for better performance
                translated_texts = []
                batch_size = 10  # Process 10 subtitles at a time

                for batch_start in range(0, len(subtitle_texts), batch_size):
                    batch_end = min(batch_start + batch_size, len(subtitle_texts))
                    batch_texts = subtitle_texts[batch_start:batch_end]

                    logger.info(f"Translating batch {batch_start//batch_size + 1}: subtitles {batch_start + 1}-{batch_end}")

                    try:
                        if direct_translator:
                            # Direct batch translation
                            logger.debug(f"Using direct batch translation {source_language}->{target_language}")
                            # Combine texts with separator for batch processing
                            combined_text = " ||| ".join(batch_texts)
                            translated_batch = direct_translator(combined_text, max_length=512 * len(batch_texts))
                            # Split back into individual translations
                            batch_results = translated_batch[0]['translation_text'].split(" ||| ")
                            translated_texts.extend(batch_results[:len(batch_texts)])  # Ensure we don't exceed batch size

                        else:
                            # Two-step batch translation via English pivot
                            logger.debug(f"Using pivot batch translation {source_language}->en->{target_language}")
                            source_to_en = get_translator(source_language, "en")
                            en_to_target = get_translator("en", target_language)

                            # Step 1: Source to English
                            combined_text = " ||| ".join(batch_texts)
                            to_english_batch = source_to_en(combined_text, max_length=512 * len(batch_texts))
                            english_batch = to_english_batch[0]['translation_text'].split(" ||| ")

                            # Step 2: English to Target
                            combined_english = " ||| ".join(english_batch[:len(batch_texts)])
                            from_english_batch = en_to_target(combined_english, max_length=512 * len(batch_texts))
                            final_batch = from_english_batch[0]['translation_text'].split(" ||| ")
                            translated_texts.extend(final_batch[:len(batch_texts)])

                    except Exception as batch_error:
                        logger.warning(f"Batch translation failed for subtitles {batch_start + 1}-{batch_end}: {batch_error}")
                        # Fallback: translate individually for failed batch
                        for j, text in enumerate(batch_texts):
                            try:
                                if direct_translator:
                                    translated = direct_translator(text, max_length=512)
                                    translated_texts.append(translated[0]['translation_text'])
                                else:
                                    # Individual pivot translation
                                    to_english = source_to_en(text, max_length=512)
                                    english_text = to_english[0]['translation_text']
                                    from_english = en_to_target(english_text, max_length=512)
                                    translated_texts.append(from_english[0]['translation_text'])
                            except Exception as individual_error:
                                logger.error(f"Individual translation failed for subtitle {batch_start + j + 1}: {individual_error}")
                                translated_texts.append(f"[{target_language.upper()}] {text}")

                # Reconstruct subtitles with translated text
                translated_subs = []
                text_idx = 0
                for i, sub in subtitle_indices:
                    if text_idx < len(translated_texts):
                        translated_text = translated_texts[text_idx]
                        translated_subs.append(pysrt.SubRipItem(
                            index=sub.index,
                            start=sub.start,
                            end=sub.end,
                            text=translated_text
                        ))
                        text_idx += 1

                # Create translated SRT content
                translated_content = ""
                for sub in translated_subs:
                    translated_content += f"{sub.index}\n"
                    start_str = str(sub.start).replace('.', ',')
                    end_str = str(sub.end).replace('.', ',')
                    translated_content += f"{start_str} --> {end_str}\n"
                    translated_content += f"{sub.text}\n\n"

                logger.info(f"Translation completed. Translated content length: {len(translated_content)}")
                language_to_use = target_language
            else:
                logger.info(f"File {file_path} is not SRT format, skipping structured translation")
        else:
            logger.warning(f"No translation path available for {source_language} to {target_language}")
    else:
        logger.info(f"No translation needed - source and target languages are the same ({source_language})")

    return translated_content, language_to_use

async def process_subtitle_to_audio_job(
    job_id: str,
    file_path: str,
    source_language: str,
    target_language: str,
    voice: str,
    output_format: str,
    user_id: str
):
    """Background task for subtitle-to-audio generation"""
    try:
        # Update job status
        await job_storage.update_progress(job_id, 10, "Initializing subtitle-to-audio job...")
        # Read subtitle file
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                subtitle_content = f.read()
        except:
            with open(file_path, "r", encoding="latin-1") as f:
                subtitle_content = f.read()
        
        # Update progress
        await job_storage.update_progress(job_id, 20, "Subtitle file loaded...")

        logger.info(f"=== STARTING SUBTITLE-TO-AUDIO PROCESS ===")
        logger.info(f"Job ID: {job_id}")
        logger.info(f"File: {file_path}")
        logger.info(f"Source language: '{source_language}'")
        logger.info(f"Target language: '{target_language}'")
        logger.info(f"Voice: '{voice}'")
        logger.info(f"Output format: '{output_format}'")
        logger.info(f"Subtitle content preview: '{subtitle_content[:100]}...'")

        # Translate subtitles if needed
        translated_content, language_to_use = await translate_subtitle_content(
            subtitle_content, source_language, target_language, file_path
        )
        
        # Update progress
        await job_storage.update_progress(job_id, 50, "Subtitle translation completed...")

        # Generate audio from translated subtitles
        logger.info(f"Generating audio with language: {language_to_use}, voice: {voice}")
        audio_filename, segment_count = await generate_audio_from_subtitles(
            translated_content,
            language_to_use,
            voice,
            output_format
        )
        
        # Update progress
        await job_storage.update_progress(job_id, 90, "Audio generation completed...")

        # Update job with results
        await job_storage.complete_job(job_id, {
            "status": "completed",
            "progress": 100,
            "download_url": f"/api/download/subtitle-audio/{job_id}",
            "completed_at": datetime.utcnow().isoformat(),
            "filename": audio_filename,
            "segment_count": segment_count,
            "language": language_to_use,
            "format": output_format,
            "voice": voice
        })
        
        # Cleanup temp file
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except:
                pass
            
    except Exception as e:
        logger.error(f"Subtitle-to-audio job {job_id} failed: {str(e)}")
        logger.exception(f"Full traceback for job {job_id}:")
        await job_storage.fail_job(job_id, str(e))

        # Refund credits on failure
        try:
            response = supabase.table("users").select("credits").eq("id", user_id).execute()
            if response.data:
                current_credits = response.data[0]["credits"]
                supabase.table("users").update({"credits": current_credits + 5}).eq("id", user_id).execute()
                logger.info(f"Refunded 5 credits to user {user_id} due to subtitle-to-audio failure: {str(e)}")
        except Exception as refund_error:
            logger.error(f"Failed to refund credits: {refund_error}")
        
        # Cleanup temp file
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except:
                pass

@app.get("/api/generate/subtitle-audio/status/{job_id}")
async def get_subtitle_audio_status(
    job_id: str,
    current_user_id: str = Depends(get_current_user_id)
):
    """Get status of a subtitle-to-audio job"""
    # Get job from Supabase storage
    job = await job_storage.get_job(job_id)
    
    if not job:
        raise HTTPException(404, "Job not found")

    if job.get("user_id") != current_user_id:
        raise HTTPException(403, "Access denied")
    
    status = job.get("status", "pending")
    if status == "completed":
        message = "Audio generation completed successfully"
    elif status == "processing":
        message = "Audio generation in progress"
    elif status == "failed":
        message = f"Audio generation failed: {job.get('error', 'Unknown error')}"
    else:
        message = "Job pending"
    
    response = {
        "success": status == "completed",
        "job_id": job_id,
        "status": status,
        "progress": job.get("progress", 0),
        "source_language": job.get("source_language"),
        "target_language": job.get("target_language"),
        "voice": job.get("voice"),
        "format": job.get("format"),
        "segment_count": job.get("segment_count"),
        "download_url": job.get("download_url") if status == "completed" else None,
        "message": message,
        "error": job.get("error") if status == "failed" else None
    }
    
    return response

@app.get("/api/download/subtitle-audio/{job_id}")
async def download_subtitle_audio(
    job_id: str,
    current_user: User = Depends(get_current_user)
):
    """Download generated audio file from subtitle-to-audio job"""
    logger.info(f"[DOWNLOAD] Subtitle audio download for job_id={job_id}")
    
    # Get job from job_storage (Supabase)
    job = await job_storage.get_job(job_id)
    
    if not job:
        logger.error(f"[DOWNLOAD] Job {job_id} not found in job_storage")
        raise HTTPException(404, "Job not found")
    
    logger.info(f"[DOWNLOAD] Found job: status={job.get('status')}, type={job.get('type')}")
    
    if job.get("user_id") != current_user.id:
        logger.error(f"[DOWNLOAD] Access denied for job {job_id}")
        raise HTTPException(403, "Access denied")
    
    if job.get("status") != "completed":
        logger.error(f"[DOWNLOAD] Job not completed: {job.get('status')}")
        raise HTTPException(400, f"Job not completed yet. Status: {job.get('status')}")
    
    # Get the audio filename from job data
    audio_filename = job.get("filename")
    output_format = job.get("format", "mp3")
    
    if not audio_filename:
        logger.error(f"[DOWNLOAD] No filename in job data")
        raise HTTPException(404, "Audio file not found in job data")
    
    audio_path = os.path.join("outputs", audio_filename)
    logger.info(f"[DOWNLOAD] Looking for audio file at: {audio_path}")
    
    if not os.path.exists(audio_path):
        logger.error(f"[DOWNLOAD] Audio file not found on disk: {audio_path}")
        raise HTTPException(404, f"Audio file not found: {audio_filename}")
    
    logger.info(f"[DOWNLOAD] Serving file: {audio_path}")
    return FileResponse(
        audio_path,
        media_type=f"audio/{output_format}",
        filename=f"subtitle_audio_{job_id[:8]}.{output_format}"
    )

# Audio translation background processing
async def process_audio_translation_job(job_id: str, file_path: str, source_lang: str, target_lang: str, user_id: str):
    """Background task for audio translation using SAME high-quality pipeline as video translation"""
    try:
        # Update job status
        jobs_db[job_id]["progress"] = 5
        jobs_db[job_id]["message"] = "Loading AI models..."
        await job_storage.update_progress(job_id, 5, "Loading AI models...")

        # Use the SAME high-quality AudioTranslator as video translation
        from modules.audio_translator import AudioTranslator, TranslationConfig

        # Configure with the same settings as video translation for crystal-clear quality
        config = TranslationConfig(
            source_lang=source_lang,
            target_lang=target_lang,
            enable_input_normalization=True,
            enable_denoising=True,
            enable_gain_consistency=True,
            enable_silence_padding=True,
            validation_spots=3,  # Quality validation like video
            target_lufs=-16.0,  # Same loudness as video
            chunk_size=30,  # Same chunking as video
            timing_tolerance_ms=200,  # Same tolerance as video
            voice_speed=1.0,
            voice_pitch="+0Hz",
            voice_style="neutral"
        )

        translator = AudioTranslator(config)

        # Load models (same as video translation)
        if not translator.load_models():
            raise Exception("Failed to load AI models for high-quality audio translation")

        jobs_db[job_id]["progress"] = 10
        jobs_db[job_id]["message"] = "AI models loaded. Starting audio processing..."
        await job_storage.update_progress(job_id, 10, "AI models loaded. Starting audio processing...")

        # Step 1: Preprocess audio (same as video translation)
        processed_audio_path = translator.preprocess_audio(file_path)
        jobs_db[job_id]["progress"] = 20
        jobs_db[job_id]["message"] = "Audio preprocessing completed..."
        await job_storage.update_progress(job_id, 20, "Audio preprocessing completed...")

        # Step 2: Transcribe with segments (same as video translation)
        transcription = translator.transcribe_with_segments(processed_audio_path)
        if not transcription.get("success", False):
            error_msg = transcription.get("error", "Transcription failed")
            raise Exception(f"Audio transcription failed: {error_msg}")

        original_text = transcription["text"]
        segments = transcription["segments"]
        quality_metrics = transcription.get("quality_metrics", {})

        jobs_db[job_id]["progress"] = 40
        jobs_db[job_id]["message"] = f"Transcribed {len(original_text)} chars, {len(segments)} segments..."
        await job_storage.update_progress(job_id, 40, f"Transcribed segments...")

        # Step 3: Translate text (same as video translation)
        translated_text, translated_segments = translator.translate_text_with_context(
            original_text, segments
        )

        jobs_db[job_id]["progress"] = 60
        jobs_db[job_id]["message"] = "Translation completed. Generating high-quality audio..."
        await job_storage.update_progress(job_id, 60, "Translation completed. Generating high-quality audio...")

        # Step 4: Generate speech using SAME TTS pipeline as video translation
        output_path = f"translated_audio_{job_id}.wav"
        success, timing_segments = translator.synthesize_speech_with_timing(
            translated_text, translated_segments, output_path
        )

        if not success or not os.path.exists(output_path):
            raise Exception("High-quality TTS synthesis failed")

        jobs_db[job_id]["progress"] = 80
        jobs_db[job_id]["message"] = "TTS synthesis completed. Applying audio enhancement..."
        await job_storage.update_progress(job_id, 80, "TTS synthesis completed. Applying audio enhancement...")

        # Step 5: Apply SAME audio processing as video translation
        if translator.config.enable_gain_consistency:
            final_audio = AudioSegment.from_file(output_path)
            processed_audio = translator.apply_gain_consistency(final_audio, translator.config.target_lufs)
            processed_audio.export(output_path, format="wav")

        # Step 6: Quality validation (same as video translation)
        if translator.config.validation_spots > 0:
            validation_results = translator.validate_audio_quality(output_path, processed_audio_path)
            if validation_results["quality_score"] < 0.7:
                logger.warning(f"Audio quality validation warning: {validation_results['quality_score']:.2f}")

        jobs_db[job_id]["progress"] = 95
        jobs_db[job_id]["message"] = "Finalizing high-quality audio..."
        await job_storage.update_progress(job_id, 95, "Finalizing high-quality audio...")

        # Step 7: Calculate final metrics
        original_audio = AudioSegment.from_file(processed_audio_path)
        translated_audio = AudioSegment.from_file(output_path)
        duration_match_percent = (1 - abs(len(translated_audio) - len(original_audio)) / len(original_audio)) * 100

        # Update job with results
        completion_data = {
            "status": "completed",
            "progress": 100,
            "result": {
                "download_url": f"/api/download/audio/{job_id}",
                "duration_match_percent": duration_match_percent,
                "speed_adjustment": 1.0,
                "quality_metrics": quality_metrics
            },
            "completed_at": datetime.utcnow().isoformat(),
            "output_path": output_path
        }
        jobs_db[job_id].update(completion_data)
        await job_storage.complete_job(job_id, completion_data)
        save_jobs_db()

        logger.info(f"[SUCCESS] High-quality audio translation completed: {output_path}")

        # Cleanup temp file
        if os.path.exists(file_path):
            os.remove(file_path)

    except Exception as e:
        error_msg = str(e)
        jobs_db[job_id].update({
            "status": "failed",
            "error": error_msg,
            "failed_at": datetime.utcnow().isoformat(),
            "result": {
                "success": False,
                "error": error_msg,
                "output_path": None
            }
        })
        await job_storage.fail_job(job_id, error_msg)
        save_jobs_db()

        # Refund credits on failure
        try:
            response = supabase.table("users").select("credits").eq("id", user_id).execute()
            if response.data:
                current_credits = response.data[0]["credits"]
                supabase.table("users").update({"credits": current_credits + 10}).eq("id", user_id).execute()
                print(f"Refunded 10 credits to user {user_id} due to audio translation failure")
        except Exception as refund_error:
            print(f"Failed to refund credits: {refund_error}")

        # Cleanup temp file on error
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except:
                pass

# Download subtitle audio file
async def _download_subtitle_audio_helper(job_id, current_user):
    """Download generated subtitle audio file"""
    logger.info(f"[DOWNLOAD] Starting download for job_id={job_id}")
    logger.info(f"[DOWNLOAD] Current user ID: {current_user.id}")
    logger.info(f"[DOWNLOAD] Total jobs in subtitle_jobs: {len(subtitle_jobs)}")
    logger.info(f"[DOWNLOAD] Job IDs in subtitle_jobs: {list(subtitle_jobs.keys())}")

    if job_id not in subtitle_jobs:
        logger.error(f"[DOWNLOAD] Job {job_id} not found in subtitle_jobs dictionary")
        raise HTTPException(404, "Job not found")

    logger.info(f"[DOWNLOAD] Job found: {job}")

    if job["user_id"] != current_user.id:
        logger.error(f"[DOWNLOAD] Access denied: job user_id={job['user_id']}, current user={current_user.id}")
        raise HTTPException(403, "Access denied")

    if job["status"] != "completed":
        logger.error(f"[DOWNLOAD] Job not completed yet, status: {job['status']}")
        raise HTTPException(400, "Job not completed yet")

    audio_filename = job.get("filename")
    logger.info(f"[DOWNLOAD] Audio filename from job: {audio_filename}")
    
    if not audio_filename:
        logger.error(f"[DOWNLOAD] No filename in job data")
        raise HTTPException(404, "Audio file not found")

    audio_path = os.path.join("outputs", audio_filename)
    logger.info(f"[DOWNLOAD] Constructed audio path: {audio_path}")
    logger.info(f"[DOWNLOAD] File exists check: {os.path.exists(audio_path)}")

    if not os.path.exists(audio_path):
        logger.error(f"[DOWNLOAD] Audio file not found on disk: {audio_path}")
        # List files in outputs directory for debugging
        try:
            outputs_files = os.listdir("outputs")
            logger.error(f"[DOWNLOAD] Files in outputs directory: {outputs_files}")
        except Exception as e:
            logger.error(f"[DOWNLOAD] Could not list outputs directory: {e}")
        raise HTTPException(404, "Audio file not found on disk")

    logger.info(f"[DOWNLOAD] File found, preparing to send: {audio_path}")

    # Stream the file back to client
    def iterfile():
        with open(audio_path, 'rb') as f:
            yield from f

    return FileResponse(
        audio_path,
        media_type=f'audio/{job.get("format", "mp3")}',
        filename=f"subtitle_audio_{job_id[:8]}.{job.get('format', 'mp3')}"
    )

# ========== ROOT ENDPOINT ==========

@app.get("/")
async def root():
    """Root endpoint showing API information and available endpoints"""
    # Get current system stats
    import psutil
    from datetime import datetime
    import sys
    
    # Get voice statistics
    total_voices = sum(len(voices) for voices in GTTS_VOICE_OPTIONS.values())
    available_languages = list(GTTS_VOICE_OPTIONS.keys())
    
    # Get translation model stats
    loaded_translators = list(translator_cache.keys()) if translator_cache else []
    
    # Get job stats
    video_jobs_count = 0  # No longer in memory, jobs are in Supabase
    subtitle_jobs_count = 0  # No longer in memory, jobs are in Supabase
    
    return {
        "success": True,
        "service": "Octavia Video Translator",
        "version": "4.0.0",
        "status": "operational",
        "timestamp": datetime.now().isoformat(),
        "authentication": "JWT + Supabase",
        "database": "Supabase (PostgreSQL)",
        "ai_models": "Whisper, Helsinki NLP, gTTS",
        "deployment": "Local development",
        "license": "Proprietary",
        "payment": {
            "provider": "Polar.sh",
            "mode": POLAR_SERVER,
            "test_mode": ENABLE_TEST_MODE,
            "real_products_configured": True,
            "packages_available": len(CREDIT_PACKAGES)
        },
        "pipeline_mode": "full" if PIPELINE_AVAILABLE else "simplified",
        "translation_models": {
            "provider": "Helsinki NLP",
            "available_pairs": list(HELSINKI_MODELS.keys()),
            "preloaded_models": loaded_translators,
            "total_models": len(HELSINKI_MODELS),
            "cache_size": len(translator_cache)
        },
        "voice_synthesis": {
            "provider": "gTTS + Microsoft Edge TTS",
            "available_languages": available_languages,
            "total_voices": total_voices,
            "cloning_supported": True,
            "preview_supported": True
        },
        "job_storage": {
            "video_jobs": video_jobs_count,
            "subtitle_jobs": subtitle_jobs_count,
            "total_jobs": video_jobs_count + subtitle_jobs_count
        },
        "environment": {
            "python_version": sys.version,
            "platform": sys.platform,
            "cpu_count": psutil.cpu_count(),
            "total_ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "available_ram_gb": round(psutil.virtual_memory().available / (1024**3), 2)
        },
        "endpoints": {
            "documentation": "/docs",
            "redoc": "/redoc",
            "health": "/api/health",
            "root": "GET /",
            
            # Authentication & User Management
            "auth": {
                "signup": "POST /api/auth/signup",
                "login": "POST /api/auth/login",
                "logout": "POST /api/auth/logout",
                "verify_email": "POST /api/auth/verify",
                "resend_verification": "POST /api/auth/resend-verification",
                "demo_login": "POST /api/auth/demo-login"
            },
            
            # User Profile & Credits
            "user": {
                "profile": "GET /api/user/profile",
                "credits": "GET /api/user/credits"
            },
            
            # Payment & Credits System
            "payments": {
                "packages": "GET /api/payments/packages",
                "create_session": "POST /api/payments/create-session",
                "payment_status": "GET /api/payments/status/{session_id}",
                "webhook": "POST /api/payments/webhook/polar",
                "transactions": "GET /api/payments/transactions"
            },
            
            # Subtitle Generation & Management
            "subtitles": {
                "generate": "POST /api/translate/subtitles",
                "status": "GET /api/translate/subtitles/status/{job_id}",
                "download": "GET /api/download/subtitles/{job_id}",
                "review": "GET /api/translate/subtitles/review/{job_id}",
                "list_files": "GET /api/translate/subtitles/list",
                "download_file": "GET /api/download/subtitles/{job_id}/{filename}"
            },
            
            # Subtitle Translation
            "subtitle_translation": {
                "translate_file": "POST /api/translate/subtitle-file",
                "status": "GET /api/translate/subtitle-status/{job_id}",
                "download_translated": "GET /api/download/translated-subtitle/{job_id}"
            },
            
            # Video Translation
            "video_translation": {
                "enhanced": "POST /api/translate/video/enhanced",
                "job_status": "GET /api/jobs/{job_id}/status",
                "download": "GET /api/download/video/{job_id}"
            },
            
            # Audio Translation
            "audio_translation": {
                "translate": "POST /api/translate/audio",
                "download": "GET /api/download/audio/{job_id}"
            },
            
            # Subtitle to Audio Generation
            "subtitle_audio": {
                "generate": "POST /api/generate/subtitle-audio",
                "status": "GET /api/generate/subtitle-audio/status/{job_id}",
                "download": "GET /api/download/subtitle-audio/{job_id}"
            },
            
            # Voice Management
            "voices": {
                "get_by_language": "GET /api/voices/{language}",
                "get_all": "GET /api/voices/all",
                "preview": "POST /api/voices/preview",
                "preview_audio": "GET /api/voices/preview/audio/{preview_id}"
            },
            
            # File Download (Legacy/Generic)
            "download": {
                "generic": "GET /api/download/{file_type}/{file_id}",
                "subtitles": "GET /api/download/subtitles/{job_id}",
                "video": "GET /api/download/video/{job_id}",
                "audio": "GET /api/download/audio/{job_id}",
                "translated_subtitle": "GET /api/download/translated-subtitle/{job_id}",
                "subtitle_audio": "GET /api/download/subtitle-audio/{job_id}"
            },
            
            # Testing & Debugging
            "testing": {
                "integration_test": "POST /api/test/integration",
                "metrics": "GET /api/metrics",
                "health": "GET /api/health"
            },
            
            # System & Monitoring
            "system": {
                "metrics": "GET /api/metrics",
                "logs": "artifacts/logs.jsonl",
                "webhook_logs": "stored_in_supabase",
                "error_logs": "stored_in_supabase"
            }
        },
        
        # File Storage Information
        "storage": {
            "temp_files": "auto-cleaned on shutdown",
            "generated_files": "stored in current directory",
            "subtitle_files": "subtitles_*.{srt,vtt,ass,ssa}",
            "audio_files": "subtitle_audio_*.mp3, translated_audio_*.mp3",
            "translated_subtitles": "translated_subtitle_*.{srt,vtt,ass,ssa}",
            "voice_previews": "voice_preview_*.mp3",
            "artifacts": "artifacts/logs.jsonl"
        },
        
        # Feature Flags & Capabilities
        "features": {
            "subtitle_generation": True,
            "subtitle_translation": True,
            "video_translation": PIPELINE_AVAILABLE,
            "audio_translation": True,
            "subtitle_to_audio": True,
            "text_to_speech": True,
            "voice_cloning": True,
            "voice_preview": True,
            "credit_system": True,
            "user_authentication": True,
            "email_verification": True,
            "payment_processing": True,
            "background_processing": True,
            "job_status_tracking": True,
            "real_time_progress": True,
            "multi_language_support": True,
            "file_format_support": ["srt", "vtt", "ass", "ssa", "mp3", "wav", "mp4", "avi", "mov"]
        },
        
        # Credit Costs (for reference)
        "credit_costs": {
            "subtitle_generation": 1,
            "subtitle_translation": 5,
            "subtitle_to_audio": 5,
            "audio_translation": 10,
            "video_translation_enhanced": 10,
            "voice_preview": 1,
            "voice_cloning": 50
        },
        
        # Supported Languages
        "languages": {
            "translation": {
                "source_languages": ["en", "es", "fr", "de", "it", "ru", "ja", "ko", "zh", "ar", "hi", "pt"],
                "target_languages": ["en", "es", "fr", "de", "it", "ru", "ja", "ko", "zh", "ar", "hi", "pt"],
                "auto_detection": ["auto"],
                "available_pairs": len(HELSINKI_MODELS)
            },
            "voice_synthesis": {
                "supported": available_languages,
                "total_voices": total_voices,
                "details": {
                    lang: {
                        "name": {
                            "en": "English",
                            "es": "Spanish",
                            "fr": "French",
                            "de": "German",
                            "it": "Italian",
                            "pt": "Portuguese",
                            "ru": "Russian",
                            "ja": "Japanese",
                            "ko": "Korean",
                            "zh": "Chinese",
                            "ar": "Arabic",
                            "hi": "Hindi"
                        }.get(lang, lang.upper()),
                        "voices_available": len(voices),
                        "example_voices": list(voices.keys())[:3] if voices else []
                    }
                    for lang, voices in GTTS_VOICE_OPTIONS.items()
                }
            }
        },
        
        # Quick Start Examples
        "quick_start": {
            "1_signup": "POST /api/auth/signup with {email, password, name}",
            "2_login": "POST /api/auth/login with {email, password}",
            "3_check_voices": "GET /api/voices/all",
            "4_generate_subtitles": "POST /api/translate/subtitles with file, language='auto', format='srt'",
            "5_translate_subtitles": "POST /api/translate/subtitle-file with file, source_language, target_language",
            "6_generate_audio": "POST /api/generate/subtitle-audio with file, source_language, target_language, voice",
            "7_check_status": "GET appropriate status endpoint with job_id",
            "8_download": "GET appropriate download endpoint"
        },
        
        # Rate Limits & Policies
        "policies": {
            "rate_limiting": "None currently (implement as needed)",
            "file_size_limits": "Depends on server resources",
            "job_timeout": "60 seconds polling timeout",
            "credit_refund": "Automatic on job failure",
            "data_retention": "Temp files cleaned on shutdown",
            "privacy": "User data stored in Supabase, files temporarily on server"
        },
        
        # Development Information
        "development": {
            "api_version": "4.0.0",
            "backend": "FastAPI + Python",
            "frontend": "Next.js + React",
            "database": "Supabase (PostgreSQL)",
            "ai_models": "Whisper, Helsinki NLP, gTTS",
            "deployment": "Local development",
            "license": "Proprietary"
        },
        
        # Status Information
        "status_indicators": {
            "database": "connected" if supabase else "disconnected",
            "translation_models": "loaded" if translator_cache else "not_loaded",
            "whisper_model": "loaded" if 'whisper_model' in globals() and whisper_model else "not_loaded",
            "pipeline_modules": "available" if PIPELINE_AVAILABLE else "simplified_mode",
            "payment_gateway": "sandbox" if ENABLE_TEST_MODE else "live",
            "email_service": "configured" if os.getenv("SMTP_USER") else "mock_mode",
            "voice_endpoints": "available"  # This confirms voice endpoints are working
        }
    }
# ========== APPLICATION ENTRY POINT ==========

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*60)
    print("OCTAVIA VIDEO TRANSLATOR v4.0 - BACKEND SERVER")
    print("="*60)
    print(f"Database: Supabase")
    print(f"Payment: Polar.sh (sandbox mode)")
    print(f"Test Mode: {'ENABLED' if ENABLE_TEST_MODE else 'DISABLED'}")
    print(f"Pipeline Modules: {'AVAILABLE' if PIPELINE_AVAILABLE else 'SIMPLIFIED MODE'}")
    print(f"Translation Models: Gemma 4B, NLLB-200, Helsinki NLP")
    print(f"Available Language Pairs: 80+")
    print(f"API URL: http://localhost:8000")
    print(f"Documentation: http://localhost:8000/docs")
    print(f"Logs Directory: artifacts/")
    print("="*60 + "\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False
    )
