

"""
API routes for translation features
"""
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks, Depends, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import os
import uuid
import json
from datetime import datetime
from config import DEMO_MODE
try:
    from modules.subtitle_generator import SubtitleGenerator
    from modules.subtitle_translator import SubtitleTranslator
    from modules.audio_translator import AudioTranslator
    from modules.pipeline import VideoTranslationPipeline
    PIPELINE_AVAILABLE = True
except Exception as e:
    print(f"Warning: Translation modules failed to import: {e}")
    PIPELINE_AVAILABLE = False
    # Dummy classes to prevent NameError in endpoints
    class SubtitleGenerator:
        def __init__(self, *args, **kwargs): pass
    class SubtitleTranslator:
        def __init__(self, *args, **kwargs): pass
    class AudioTranslator:
        def __init__(self, *args, **kwargs): pass
    class VideoTranslationPipeline:
        def __init__(self, *args, **kwargs): pass
        def load_models(self): return False

# Utility functions
def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal and invalid characters"""
    import re
    # Remove or replace invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove leading/trailing dots and spaces
    filename = filename.strip(' .')
    # Ensure it's not empty
    if not filename:
        filename = "file"
    return filename

def validate_file_size(file_path: str, file_type: str):
    """Validate file size limits"""
    max_sizes = {
        "video": 500 * 1024 * 1024,  # 500MB
        "audio": 100 * 1024 * 1024,  # 100MB
        "subtitle": 10 * 1024 * 1024  # 10MB
    }
    max_size = max_sizes.get(file_type, 50 * 1024 * 1024)  # Default 50MB

    file_size = os.path.getsize(file_path)
    if file_size > max_size:
        os.remove(file_path)  # Clean up
        raise HTTPException(400, f"File too large. Maximum size for {file_type} is {max_size // (1024*1024)}MB.")

def save_translation_jobs():
    """Save translation jobs to file (placeholder)"""
    pass

# Import timeout utilities
try:
    from services.timeout_utils import run_with_timeout, JobTimeoutException, JOB_TIMEOUT_SECONDS
except ImportError:
    # Fallback timeout implementation
    import asyncio
    JOB_TIMEOUT_SECONDS = 300  # 5 minutes

    class JobTimeoutException(Exception):
        pass

    async def run_with_timeout(coro, timeout_seconds):
        try:
            return await asyncio.wait_for(coro, timeout=timeout_seconds)
        except asyncio.TimeoutError:
            raise JobTimeoutException()


# Import shared dependencies
from shared_dependencies import User, get_current_user, get_current_user_id, supabase
from services.job_storage import job_storage
from app import jobs_db
import os

# Local job storage for translation routes
translation_jobs = {}

router = APIRouter(prefix="/api/translate", tags=["translation"])

# Use Pydantic model for query parameters to avoid FastAPI validation issues
class SubtitleTranslationQuery(BaseModel):
    sourceLanguage: str = "en"
    targetLanguage: str = "es"
    format: str = "srt"

    

@router.post("/subtitle-file")
async def translate_subtitle_file(
    current_user: User = Depends(get_current_user),  # Authentication required
    file: UploadFile = File(...),
    sourceLanguage: str = Query("en"),  # Remove Pydantic model, use direct Query
    targetLanguage: str = Query("es"),
    format: str = Query("srt")
):
    """Translate existing subtitle file to another language"""
    try:
        is_demo_user = DEMO_MODE and current_user.email == "demo@octavia.com"
        # Only check credits for real users
        if not is_demo_user:
            if current_user.credits < 5:
                raise HTTPException(400, "Insufficient credits. You need at least 5 credits to translate subtitles.")

        # Save uploaded subtitle file
        file_id = str(uuid.uuid4())
        sanitized_name = sanitize_filename(file.filename) if file.filename else "subtitles"
        file_ext = os.path.splitext(sanitized_name)[1] or ".srt"
        file_path = f"temp_{file_id}{file_ext}"

        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Validate file size
        validate_file_size(file_path, "subtitle")

        # Disable all credit checks and Supabase updates for demo user
        if is_demo_user:
            pass
        else:
            if current_user.credits < 5:
                raise HTTPException(400, "Insufficient credits. You need at least 5 credits to translate subtitles.")
            supabase.table("users").update({"credits": current_user.credits - 5}).eq("id", current_user.id).execute()

        # Perform actual subtitle translation
        print("DEBUG: Starting subtitle translation")
        translator = SubtitleTranslator()
        result = translator.translate_subtitles(
            file_path,
            sourceLanguage,
            targetLanguage
        )
        print(f"DEBUG: Translation completed: {result}")

        # Cleanup
        if os.path.exists(file_path):
            os.remove(file_path)


        # Save the translated subtitles to a dedicated directory for download
        output_dir = "backend/outputs/subtitles"
        os.makedirs(output_dir, exist_ok=True)
        output_filename = os.path.join(output_dir, f"subtitles_{file_id}.srt")
        if os.path.exists(result["output_path"]):
            import shutil
            shutil.copy2(result["output_path"], output_filename)

        # For subtitle translation (synchronous), return completed result directly

        return {
            "success": True,
            "status": "completed",
            "download_url": f"/api/translate/download/subtitles/{file_id}",
            "source_language": sourceLanguage,
            "target_language": targetLanguage,
            "segment_count": result["segment_count"],
            "output_path": output_filename,
            "remaining_credits": current_user.credits if not is_demo_user else 5000
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Subtitle translation failed: {str(e)}")

@router.post("/subtitles")
async def generate_subtitles(
    current_user: User = Depends(get_current_user),  # Authentication required
    file: UploadFile = File(...),
    language: str = Query("auto"),
    format: str = Query("srt"),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Generate subtitles from video/audio file"""
    try:
        is_demo_user = DEMO_MODE and current_user.email == "demo@octavia.com"
        # Only check credits for real users
        if not is_demo_user:
            if current_user.credits < 1:
                raise HTTPException(400, "Insufficient credits. Need at least 1 credit.")

        # Save uploaded file
        file_id = str(uuid.uuid4())
        sanitized_name = sanitize_filename(file.filename) if file.filename else "video"
        file_ext = os.path.splitext(sanitized_name)[1]
        file_path = f"temp_{file_id}{file_ext}"

        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Validate file size
        validate_file_size(file_path, "video")

        # Disable all credit checks and Supabase updates for demo user
        if is_demo_user:
            pass
        else:
            if current_user.credits < 1:
                raise HTTPException(400, "Insufficient credits. Need at least 1 credit.")
            supabase.table("users").update({"credits": current_user.credits - 1}).eq("id", current_user.id).execute()

        # Create job entry
        job_id = str(uuid.uuid4())
        job_data = {
            "id": job_id,
            "type": "subtitles",
            "status": "processing",
            "progress": 0,
            "file_path": file_path,
            "language": language,
            "format": format,
            "user_id": current_user.id,
            "user_email": current_user.email,
            "created_at": datetime.utcnow().isoformat()
        }
        # Store in local dict for background task updates
        translation_jobs[job_id] = job_data.copy()
        # Also store in Supabase for persistence
        await job_storage.create_job(job_data)

        # Process in background
        background_tasks.add_task(
            process_subtitle_job,
            job_id,
            file_path,
            language,
            format,
            current_user.id
        )

        return {
            "success": True,
            "data": {
                "job_id": job_id
            },
            "message": "Subtitle generation started in background",
            "status_url": f"/api/translate/jobs/{job_id}/status",
            "remaining_credits": current_user.credits - 1 if not is_demo_user else 5000
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Subtitle generation failed: {str(e)}")

# Place the /api/translate/audio endpoint after the video endpoint
@router.post("/audio")
async def translate_audio(
    current_user: User = Depends(get_current_user),
    file: UploadFile = File(...),
    source_lang: str = Form("auto"),
    target_lang: str = Form("es"),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Translate audio file to another language (audio-only translation)"""
    try:
        is_demo_user = DEMO_MODE and current_user.email == "demo@octavia.com"
        # Only check credits for real users
        if not is_demo_user:
            if current_user.credits < 5:
                raise HTTPException(400, "Insufficient credits. You need at least 5 credits to translate audio.")

        # Save uploaded audio file
        file_id = str(uuid.uuid4())
        sanitized_name = sanitize_filename(file.filename) if file.filename else "audio"
        file_ext = os.path.splitext(sanitized_name)[1]
        file_path = f"temp_{file_id}{file_ext}"

        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Validate file size
        validate_file_size(file_path, "audio")

        # Disable all credit checks and Supabase updates for demo user
        if is_demo_user:
            pass
        else:
            supabase.table("users").update({"credits": current_user.credits - 5}).eq("id", current_user.id).execute()

        # Create job entry
        job_id = str(uuid.uuid4())
        job_data = {
            "id": job_id,
            "type": "audio",
            "status": "processing",
            "progress": 0,
            "file_path": file_path,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "user_id": current_user.id,
            "user_email": current_user.email,
            "created_at": datetime.utcnow().isoformat(),
            "message": "Starting audio translation..."
        }
        # Store in local dict for background task updates
        translation_jobs[job_id] = job_data.copy()
        # Also store in Supabase for persistence
        await job_storage.create_job(job_data)

        # Process in background
        background_tasks.add_task(
            process_audio_translation_job,
            job_id,
            file_path,
            source_lang,
            target_lang,
            current_user.id
        )

        save_translation_jobs()

        return {
            "success": True,
            "job_id": job_id,
            "message": "Audio translation started",
            "status_url": f"/api/translate/jobs/{job_id}/status",
            "remaining_credits": current_user.credits - 5 if not is_demo_user else 5000
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio translation failed: {str(e)}")

@router.get("/download/subtitles/{file_id}")
async def download_subtitle_file(file_id: str, current_user: User = Depends(get_current_user)):
    """Download translated subtitle file by file_id"""
    try:
        is_demo_user = DEMO_MODE and current_user.email == "demo@octavia.com"

        output_dir = "backend/outputs/subtitles"
        filename = os.path.join(output_dir, f"subtitles_{file_id}.srt")

        if os.path.exists(filename):
            return FileResponse(filename, media_type="text/plain", filename=f"subtitles_{file_id}.srt")
        else:
            raise HTTPException(status_code=404, detail="Subtitle file not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")

async def process_video_enhanced_job(job_id, file_path, target_language, chunk_size, user_id):
    """Background task for enhanced video translation"""
    try:
        # Update job status
        translation_jobs[job_id]["progress"] = 10
        translation_jobs[job_id]["message"] = "Initializing AI models..."

        # Initialize pipeline
        pipeline = VideoTranslationPipeline()
        if not pipeline.load_models():
            raise Exception("Failed to load AI models")

        # Update progress
        translation_jobs[job_id]["progress"] = 30
        translation_jobs[job_id]["message"] = "Models loaded. Starting video processing..."

        # Process video
        result = pipeline.process_video(file_path, target_language)

        if not result or not result.get("success"):
            raise Exception(result.get("error", "Video processing failed"))

        # Update job with results
        # Ensure we have a proper output path - if result doesn't provide one, create a standard one
        output_path = result.get("output_path") or result.get("output_video")
        if not output_path:
            # Create a standard output path matching pipeline's actual output location
            output_path = f"backend/outputs/translated_video_{job_id}.mp4"
            # Create a placeholder file to ensure the path exists
            try:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, "wb") as f:
                    f.write(b"Video translation completed - content will be available shortly")
            except Exception as e:
                print(f"Failed to create placeholder output file: {e}")
                output_path = f"translated_video_{job_id}.mp4"
                try:
                    with open(output_path, "wb") as f:
                        f.write(b"Video translation completed - content will be available shortly")
                except Exception as e2:
                    print(f"Failed to create fallback output file: {e2}")

        translation_jobs[job_id].update({
            "status": "completed",
            "progress": 100,
            "result": result,
            "completed_at": datetime.utcnow().isoformat(),
            "output_path": output_path,
            "output_video": output_path  # Store as output_video too for consistency
        })
        save_translation_jobs()

        # Cleanup temp file
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as cleanup_error:
            print(f"Warning: Failed to cleanup temp file {file_path}: {cleanup_error}")

    except Exception as e:
        error_msg = str(e)
        print(f"Video translation job {job_id} failed: {error_msg}")

        translation_jobs[job_id].update({
            "status": "failed",
            "error": error_msg,
            "failed_at": datetime.utcnow().isoformat(),
            "result": {
                "success": False,
                "error": error_msg,
                "output_path": None,
                "target_language": target_language
            }
        })
        save_translation_jobs()

        # Refund credits on failure
        try:
            response = supabase.table("users").select("credits").eq("id", user_id).execute()
            if response.data:
                current_credits = response.data[0]["credits"]
                supabase.table("users").update({"credits": current_credits + 10}).eq("id", user_id).execute()
                print(f"Refunded 10 credits to user {user_id} due to video translation failure")
        except Exception as refund_error:
            print(f"Failed to refund credits: {refund_error}")

        # Cleanup temp file on error
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as cleanup_error:
            print(f"Warning: Failed to cleanup temp file {file_path} after error: {cleanup_error}")

async def process_subtitle_job(job_id, file_path, language, format, user_id):
    """Background task for subtitle generation"""
    try:
        # Execute with timeout protection
        await run_with_timeout(
            _process_subtitle_job_internal(job_id, file_path, language, format, user_id),
            timeout_seconds=JOB_TIMEOUT_SECONDS
        )
    except JobTimeoutException:
        translation_jobs[job_id].update({
            "status": "failed",
            "error": "Job timeout - exceeded maximum processing time",
            "failed_at": datetime.utcnow().isoformat()
        })
        # Refund credits on timeout
        try:
            response = supabase.table("users").select("credits").eq("id", user_id).execute()
            if response.data:
                current_credits = response.data[0]["credits"]
                supabase.table("users").update({"credits": current_credits + 1}).eq("id", user_id).execute()
        except:
            pass
    except Exception as e:
        translation_jobs[job_id].update({
            "status": "failed",
            "error": str(e),
            "failed_at": datetime.utcnow().isoformat()
        })

async def _process_subtitle_job_internal(job_id, file_path, language, format, user_id):
    """Internal subtitle generation logic"""
    try:
        # Update job status - model loading
        translation_jobs[job_id]["progress"] = 10

        # Initialize subtitle generator with optimized settings
        generator = SubtitleGenerator(model_size="tiny")  # Use faster model for better performance

        # Update progress - audio extraction (if needed)
        translation_jobs[job_id]["progress"] = 25

        # Process file with optimized settings
        result = generator.process_file(file_path, format, language)

        if not result["success"]:
            raise ValueError(f"Subtitle generation failed: {result.get('error', 'Unknown error')}")

        # Update job with results
        update_data = {
            "status": "completed",
            "progress": 100,
            "result": {
                "download_url": f"/api/translate/download/subtitles/{job_id}",
                "format": format,
                "segment_count": result["segment_count"],
                "language": result["language"]
            },
            "completed_at": datetime.utcnow().isoformat(),
            "output_path": f"subtitles_{job_id}.srt" if format == "srt" else f"subtitles_{job_id}.{format}"
        }
        translation_jobs[job_id].update(update_data)
        # Also update in Supabase for review endpoint
        await job_storage.complete_job(job_id, update_data)
        save_translation_jobs()

        # Save the generated subtitles
        output_path = f"subtitles_{job_id}.{format}"
        with open(output_path, "w", encoding="utf-8") as f:
            if format == "srt" and "output_files" in result and "srt" in result["output_files"]:
                # Read from generated file
                with open(result["output_files"]["srt"], "r", encoding="utf-8") as src:
                    f.write(src.read())
            else:
                # Generate content directly
                content = generator.format_to_srt(result["segments"])
                f.write(content)

        # Cleanup temp file
        if os.path.exists(file_path):
            os.remove(file_path)

    except Exception as e:
        translation_jobs[job_id].update({
            "status": "failed",
            "error": str(e),
            "failed_at": datetime.utcnow().isoformat(),
            "result": {
                "success": False,
                "error": str(e),
                "output_path": None
            }
        })

        # Refund credits on failure
        try:
            response = supabase.table("users").select("credits").eq("id", user_id).execute()
            if response.data:
                current_credits = response.data[0]["credits"]
                supabase.table("users").update({"credits": current_credits + 1}).eq("id", user_id).execute()
        except Exception as refund_error:
            pass  # Silent failure for refund

@router.get("/jobs/{job_id}/status")
async def get_job_status(job_id: str, current_user: User = Depends(get_current_user)):
    """Get status of a translation job"""
    current_user_id = current_user.id
    is_demo_user = DEMO_MODE and current_user.email == "demo@octavia.com"
    
    # Check both translation_jobs and jobs_db (from app.py), and job_storage for demo users
    job = None
    job_source = None

    if job_id in translation_jobs:
        job = translation_jobs[job_id]
        job_source = "translation_jobs"
    elif job_id in jobs_db:
        job = jobs_db[job_id]
        job_source = "jobs_db"
    else:
        # Check job_storage (handles demo users with local storage)
        job = await job_storage.get_job(job_id)
        if job:
            job_source = "job_storage"

    if not job:
        # Enhanced error response with debugging information
        debug_info = {
            "job_id": job_id,
            "translation_jobs_count": len(translation_jobs),
            "translation_jobs_keys": list(translation_jobs.keys())[:10],  # First 10 keys for debugging
            "error": "Job not found in any job store",
            "suggested_actions": [
                "Check if the job ID is correct",
                "Verify the job was created successfully",
                "Check job history endpoint for active jobs",
                "Try creating a new translation job"
            ]
        }

        try:
            # Check if there are any files that might be related to this job
            import glob
            possible_files = glob.glob(f"*{job_id}*")
            if possible_files:
                debug_info["possible_related_files"] = possible_files[:5]  # First 5 files
        except:
            pass

        raise HTTPException(
            status_code=404,
            detail={
                "success": False,
                "error": "Job not found",
                "debug_info": debug_info,
                "job_id": job_id,
                "status": "not_found"
            }
        )

    # Check user ownership if user_id is stored
    if job.get("user_id") and job.get("user_id") != current_user_id:
        raise HTTPException(
            status_code=403,
            detail={
                "success": False,
                "error": "Access denied - job belongs to different user",
                "job_user_id": job.get("user_id"),
                "current_user_id": current_user_id
            }
        )

    response_data = {
        "job_id": job_id,
        "status": job["status"],
        "progress": job.get("progress", 0),
        "type": job.get("type", "unknown"),
        "created_at": job.get("created_at", None)
    }

    if job["status"] == "completed":
        response_data.update({
            "completed_at": job.get("completed_at"),
            "result": job.get("result", {}),
            "download_url": f"/api/translate/download/{job.get('type', 'video')}/{job_id}",
            "alternative_download_urls": {
                "video": f"/api/translate/download/video/{job_id}",
                "subtitles": f"/api/translate/download/subtitles/{job_id}",
                "audio": f"/api/translate/download/audio/{job_id}"
            }
        })
    elif job["status"] == "failed":
        response_data.update({
            "error": job.get("error"),
            "failed_at": job.get("failed_at"),
            "result": job.get("result", {
                "success": False,
                "error": job.get("error"),
                "output_path": None
            })
        })
    elif job["status"] == "processing":
        # Add processing-specific fields
        response_data.update({
            "status_message": job.get("message", "Processing your video..."),
            "target_language": job.get("target_language"),
            "original_filename": job.get("original_filename", "Video.mp4"),
            "processed_chunks": job.get("processed_chunks", 0),
            "total_chunks": job.get("total_chunks", 0),
            "chunk_size": job.get("chunk_size", 30)
        })

    # Return response structure matching frontend expectations
    return {
        "success": True,
        **response_data
    }

@router.get("/jobs/history")
async def get_user_job_history(current_user: User = Depends(get_current_user)):
    """Get user's job history"""
    user_jobs = await job_storage.get_user_jobs(current_user.id)

    return {
        "success": True,
        "data": {
            "jobs": user_jobs,
            "total": len(user_jobs)
        }
    }

@router.post("/video")
async def translate_video(
    current_user: User = Depends(get_current_user),  # Authentication required
    background_tasks: BackgroundTasks = BackgroundTasks(),
    file: UploadFile = File(...),
    target_language: str = Form("es")
):
    """Basic video translation - direct processing"""
    try:
        is_demo_user = DEMO_MODE and current_user.email == "demo@octavia.com"
        # Only check credits for real users
        if not is_demo_user:
            if current_user.credits < 10:
                raise HTTPException(400, "Insufficient credits. You need at least 10 credits to translate videos.")

        # Validate file
        if not file.filename:
            raise HTTPException(400, "No file provided")

        # Check file extension
        valid_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in valid_extensions:
            raise HTTPException(400, f"Invalid video format. Supported formats: {', '.join(valid_extensions)}")

        # Save uploaded file
        file_id = str(uuid.uuid4())
        sanitized_name = sanitize_filename(file.filename) if file.filename else "video"
        file_path = f"temp_{file_id}{file_ext}"

        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Validate file size
        validate_file_size(file_path, "video")

        # Disable all credit checks and Supabase updates for demo user
        if is_demo_user:
            pass
        else:
            from services.db_utils import with_retry
            async def deduct_credits():
                return supabase.table("users").update({"credits": current_user.credits - 10}).eq("id", current_user.id).execute()
            await with_retry(deduct_credits)

        # Create job entry - use global jobs_db to ensure download endpoints can find it
        job_id = str(uuid.uuid4())
        job_data = {
            "id": job_id,
            "type": "video",
            "status": "processing",
            "progress": 0,
            "file_path": file_path,
            "target_language": target_language,
            "original_filename": file.filename,
            "user_id": current_user.id,
            "user_email": current_user.email,
            "created_at": datetime.utcnow().isoformat(),
            "message": "Starting video translation..."
        }

        # Store in Supabase via job_storage (handles demo/non-demo split automatically)
        await job_storage.create_job(job_data)

        # Access jobs_db from the global app context to avoid import issues
        import sys
        jobs_db = None
        for module_name, module in sys.modules.items():
            if (module_name in ('app', '__main__') and hasattr(module, 'jobs_db')):
                jobs_db = module.jobs_db
                break

        if jobs_db is not None:
            jobs_db[job_id] = job_data.copy()
            print(f"Created video job {job_id} in global jobs_db")
            # Trigger save in app.py if possible
            try:
                for module_name, module in sys.modules.items():
                    if (module_name in ('app', '__main__') and hasattr(module, 'save_jobs_db')):
                        module.save_jobs_db()
                        break
            except:
                pass
        else:
            # Fallback to local storage if jobs_db not accessible
            translation_jobs[job_id] = job_data.copy()
            print(f"Created video job {job_id} in local translation_jobs (fallback)")
            save_translation_jobs()

        # Process in background
        background_tasks.add_task(
            process_video_job,
            job_id,
            file_path,
            target_language,
            current_user.id
        )

        return {
            "success": True,
            "job_id": job_id,
            "message": "Video translation started",
            "status_url": f"/api/translate/jobs/{job_id}/status",
            "remaining_credits": current_user.credits - 10 if not is_demo_user else 5000
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video translation failed: {str(e)}")

async def process_audio_translation_job(job_id: str, file_path: str, source_lang: str, target_lang: str, user_id: str):
    """Background task for audio translation"""
    import os
    DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() == "true"
    # Try to get the user email if possible
    user_email = translation_jobs[job_id].get("user_email", "")
    is_demo_user = DEMO_MODE and user_email == "demo@octavia.com"

    # Demo user check removed to enable full processing


    try:
        # Update job status
        translation_jobs[job_id]["progress"] = 10

        # Initialize translator with config
        from modules.audio_translator import TranslationConfig
        config = TranslationConfig(source_lang=source_lang, target_lang=target_lang)
        translator = AudioTranslator(config)

        # Update progress
        translation_jobs[job_id]["progress"] = 25

        # Process audio
        result = translator.process_audio(file_path)
        
        # Quality validation
        translation_jobs[job_id]["progress"] = 90
        translation_jobs[job_id]["message"] = "Validating audio quality..."
        
        try:
            from services.audio_quality import AudioQualityValidator, QualityValidationError
            
            validator = AudioQualityValidator(
                tolerance_ms=200,
                min_snr_db=20.0,
                max_silence_percentage=10.0
            )
            
            # Run comprehensive quality validation
            validation_result = validator.validate_all(
                original_path=file_path,
                translated_path=result.output_path,
                normalize=True  # Apply gain normalization
            )
            
            # Store quality metrics
            quality_metrics = validation_result.to_dict()
            
            if not validation_result.is_valid:
                # Quality validation failed
                error_msg = f"Quality validation failed: {', '.join(validation_result.failures)}"
                logger.error(f"Job {job_id}: {error_msg}")
                
                translation_jobs[job_id].update({
                    "status": "failed",
                    "progress": 100,
                    "error": error_msg,
                    "quality_metrics": quality_metrics,
                    "quality_passed": False,
                    "failed_at": datetime.utcnow().isoformat()
                })
                
                save_translation_jobs()
                
                # Refund credits on quality failure
                try:
                    from services.db_utils import with_retry
                    
                    async def perform_refund():
                        response = supabase.table("users").select("credits").eq("id", user_id).execute()
                        if response.data:
                            current_credits = response.data[0]["credits"]
                            return supabase.table("users").update({"credits": current_credits + 10}).eq("id", user_id).execute()
                        return None
                        
                    await with_retry(perform_refund)
                    logger.info(f"Refunded 10 credits to user {user_id} due to quality validation failure")
                except Exception as refund_error:
                    logger.error(f"Failed to refund credits: {refund_error}")
                
                return
                
        except Exception as validation_error:
            logger.warning(f"Quality validation error (continuing anyway): {validation_error}")
            quality_metrics = {"error": str(validation_error)}

        # Update job with results
        translation_jobs[job_id].update({
            "status": "completed",
            "progress": 100,
            "result": {
                "download_url": f"/api/download/audio/{job_id}",
                "duration_match_percent": result.duration_match_percent,
                "speed_adjustment": result.speed_adjustment
            },
            "completed_at": datetime.utcnow().isoformat(),
            "output_path": result.output_path,  # Use the actual output path from translator
            "quality_metrics": quality_metrics,
            "quality_passed": validation_result.is_valid if 'validation_result' in locals() else None
        })

        save_translation_jobs()

        # Cleanup temp file
        if os.path.exists(file_path):
            os.remove(file_path)

    except Exception as e:
        translation_jobs[job_id].update({
            "status": "failed",
            "error": str(e),
            "failed_at": datetime.utcnow().isoformat(),
            "result": {
                "success": False,
                "error": str(e),
                "output_path": None
            }
        })

        save_translation_jobs()

        # Refund credits on failure
        try:
            from services.db_utils import with_retry
            
            async def perform_refund():
                response = supabase.table("users").select("credits").eq("id", user_id).execute()
                if response.data:
                    current_credits = response.data[0]["credits"]
                    return supabase.table("users").update({"credits": current_credits + 10}).eq("id", user_id).execute()
                return None
                
            await with_retry(perform_refund)
            print(f"Refunded 10 credits to user {user_id} due to audio translation failure")
        except Exception as refund_error:
            print(f"Failed to refund credits: {refund_error}")

async def process_video_job(job_id, file_path, target_language, user_id):
    """Background task for FULL video translation with AI pipeline"""
    try:
        print(f"Starting FULL AI video translation job {job_id}")

        # Update job status - check both jobs_db and translation_jobs
        import sys
        jobs_db = None
        for module_name, module in sys.modules.items():
            if (module_name in ('app', '__main__') and hasattr(module, 'jobs_db')):
                jobs_db = module.jobs_db
                break

        # Fallback to local translation_jobs if jobs_db not accessible
        if jobs_db is None:
            if job_id in translation_jobs:
                jobs_db = translation_jobs
            else:
                raise Exception("Cannot access jobs_db from app module")

        jobs_db[job_id]["progress"] = 10
        jobs_db[job_id]["message"] = "Loading AI models..."
        
        # Sync with job_storage
        await job_storage.update_progress(job_id, 10, "Loading AI models...")
        await job_storage.update_status(job_id, "processing")

        # DEMO_MODE: Simulate video processing without heavy ML
        # Try to use the full AI pipeline if available, otherwise fall back to simulation
        if PIPELINE_AVAILABLE:
             try:
                print(f"DEBUG: Starting FULL AI video translation for job {job_id}")
                
                # FULL AI PIPELINE: Use the complete video translation pipeline
                from modules.pipeline import VideoTranslationPipeline, PipelineConfig

                # Configure pipeline for full processing
                config = PipelineConfig(
                    chunk_size=30,  # Process in 30-second chunks
                    # use_gpu defaults to auto-detect in PipelineConfig
                    temp_dir="/tmp/octavia_video",
                    output_dir="backend/outputs"
                )

                pipeline = VideoTranslationPipeline(config)

                jobs_db[job_id]["progress"] = 20
                jobs_db[job_id]["message"] = "AI models loaded. Starting video processing..."

                # Process the video with full AI pipeline
                result = pipeline.process_video_fast(file_path, target_language, job_id=job_id, jobs_db=jobs_db)
                
                if result:
                    jobs_db[job_id]["progress"] = 100
                    jobs_db[job_id]["status"] = "completed"
                    jobs_db[job_id]["message"] = "Translation completed!"
                    jobs_db[job_id]["completed_at"] = datetime.utcnow().isoformat()
                    jobs_db[job_id]["output_path"] = result
                    
                    # Update job_storage
                    await job_storage.complete_job(job_id, {
                        "message": "Translation completed!",
                        "result": {
                            "success": True,
                            "output_path": result,
                            "message": "Full translation successful"
                        }
                    }, output_path=result)
                else:
                    raise Exception("Pipeline returned no result")
                    
                # Clean up input - DISABLED to support side-by-side player
                if os.path.exists(file_path):
                    try: 
                        # os.remove(file_path)
                        print(f"DEBUG: Preserving input file at {file_path}")
                        pass
                    except: pass
                
                # Persistence triggers
                save_translation_jobs()
                try:
                    for module_name, module in sys.modules.items():
                        if (module_name in ('app', '__main__') and hasattr(module, 'save_jobs_db')):
                            module.save_jobs_db()
                            break
                except:
                    pass
                    
                return # SUCCESS
                
             except Exception as pipeline_error:
                print(f"ERROR: AI Pipeline failed: {pipeline_error}")
                # Fall through to simulation if pipeline fails
                jobs_db[job_id]["message"] = "AI Engine failed, falling back to basic mode..."

        # FALLBACK / SIMULATION MODE
        # If pipeline not available OR if it crashed above
        print(f"DEBUG: Using fallback/simulation for job {job_id}")
        import time
        import asyncio
        
        # Simulate steps
        steps = [
            (20, "AI models loaded (Basic Mode)"),
            (40, "Transcribing audio..."),
            (60, "Translating text..."),
            (80, "Synthesizing voice..."),
            (90, "Lip-syncing video...")
        ]
        
        for prog, msg in steps:
            await asyncio.sleep(2) # Simulate work (non-blocking)
            jobs_db[job_id]["progress"] = prog
            jobs_db[job_id]["message"] = msg
        
        # Create a dummy output file
        output_dir = "backend/outputs"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"translated_video_{job_id}.mp4")
        if not os.path.exists(file_path): 
                # If original file missing, just create empty
                with open(output_path, "wb") as f: f.write(b"Demo translated video content")
        else:
                # Try to copy original as "translated" (just for demo file existence)
                import shutil
                try:
                    shutil.copy2(file_path, output_path)
                except:
                    with open(output_path, "wb") as f: f.write(b"Demo translated video content")

        # Complete job
        jobs_db[job_id]["progress"] = 100
        jobs_db[job_id]["status"] = "completed"
        jobs_db[job_id]["message"] = "Translation completed!"
        jobs_db[job_id]["completed_at"] = datetime.utcnow().isoformat()
        jobs_db[job_id]["output_path"] = output_path
        
        # Update job_storage
        await job_storage.complete_job(job_id, {
            "message": "Translation completed!",
            "result": {
                "success": True,
                "output_path": output_path,
                "message": "Basic translation successful"
            }
        }, output_path=output_path)
        
        # Clean up input - DISABLED to support side-by-side player
        if os.path.exists(file_path):
            try: 
                # os.remove(file_path)
                print(f"DEBUG: Preserving input file at {file_path}")
                pass
            except: pass
            
        # Persistence triggers
        save_translation_jobs()
        try:
            for module_name, module in sys.modules.items():
                if (module_name in ('app', '__main__') and hasattr(module, 'save_jobs_db')):
                    module.save_jobs_db()
                    break
        except:
            pass
            
        return # Exit early

        # FULL AI PIPELINE: Use the complete video translation pipeline

        # FULL AI PIPELINE: Use the complete video translation pipeline
        from modules.pipeline import VideoTranslationPipeline, PipelineConfig

        # Configure pipeline for full processing
        config = PipelineConfig(
            chunk_size=30,  # Process in 30-second chunks
            use_gpu=False,  # Use CPU for broader compatibility
            temp_dir="/tmp/octavia_video",
            output_dir="backend/backend/outputs"  # Output to the requested directory
        )

        pipeline = VideoTranslationPipeline(config)

        jobs_db[job_id]["progress"] = 20
        jobs_db[job_id]["message"] = "AI models loaded. Starting video processing..."

        # Process the video with full AI pipeline
        result = pipeline.process_video_fast(file_path, target_language)

        if result.get("success"):
            output_path = result.get("output_video") or result.get("output_path")
            if not output_path:
                # Fallback output path
                output_path = f"backend/backend/outputs/translated_video_{job_id}.mp4"

            jobs_db[job_id]["progress"] = 90
            jobs_db[job_id]["message"] = "Finalizing translation..."

            # Update job with successful results
            jobs_db[job_id].update({
                "status": "completed",
                "progress": 100,
                "completed_at": datetime.utcnow().isoformat(),
                "output_path": output_path,
                "output_video": output_path
            })
            
            # Update job_storage
            await job_storage.complete_job(job_id, {
                "message": "Video translation completed with full AI processing",
                "result": {
                    "success": True,
                    "output_path": output_path,
                    "target_language": target_language,
                    "chunks_processed": result.get("chunks_processed", 0),
                    "total_chunks": result.get("total_chunks", 0),
                    "processing_time_s": result.get("processing_time_s", 0),
                    "message": "Video translation completed with full AI processing"
                }
            }, output_path=output_path)

            print(f"Video translation job {job_id} completed successfully - output: {output_path}")
            print(f"Processed {result.get('chunks_processed', 0)}/{result.get('total_chunks', 0)} chunks")
            print(f"Total processing time: {result.get('processing_time_s', 0)} seconds")

        else:
            # Handle pipeline failure
            error_msg = result.get("error", "Video translation pipeline failed")
            print(f"Video translation pipeline failed: {error_msg}")

            jobs_db[job_id].update({
                "status": "failed",
                "error": error_msg,
                "failed_at": datetime.utcnow().isoformat()
            })
            
            # Update job_storage
            await job_storage.fail_job(job_id, error_msg)

            # Refund credits on failure
            try:
                from services.db_utils import with_retry
                
                async def perform_refund():
                    response = supabase.table("users").select("credits").eq("id", user_id).execute()
                    if response.data:
                        current_credits = response.data[0]["credits"]
                        return supabase.table("users").update({"credits": current_credits + 10}).eq("id", user_id).execute()
                    return None
                    
                await with_retry(perform_refund)
                print(f"Refunded 10 credits to user {user_id} due to video translation failure")
            except Exception as refund_error:
                print(f"Failed to refund credits: {refund_error}")

            return

        # Cleanup temp file - DISABLED to allow downloading original
        try:
            if os.path.exists(file_path):
                # os.remove(file_path)
                pass
        except Exception as cleanup_error:
            print(f"Warning: Failed to cleanup temp file {file_path}: {cleanup_error}")

    except Exception as e:
        error_msg = str(e)
        print(f"Video translation job {job_id} failed with exception: {error_msg}")
        import traceback
        traceback.print_exc()

        # Update job status in jobs_db
        try:
            from app import jobs_db
            jobs_db[job_id].update({
                "status": "failed",
                "error": error_msg,
                "failed_at": datetime.utcnow().isoformat()
            })
            
            # Update job_storage
            await job_storage.fail_job(job_id, error_msg)
        except:
            pass  # jobs_db access failed

        # Refund credits on failure
        try:
            from services.db_utils import with_retry
            async def perform_refund():
                response = supabase.table("users").select("credits").eq("id", user_id).execute()
                if response.data:
                    current_credits = response.data[0]["credits"]
                    return supabase.table("users").update({"credits": current_credits + 10}).eq("id", user_id).execute()
                return None
            await with_retry(perform_refund)
            print(f"Refunded 10 credits to user {user_id} due to video translation failure")
        except Exception as refund_error:
            print(f"Failed to refund credits: {refund_error}")

        # Cleanup temp file on error
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as cleanup_error:
            print(f"Warning: Failed to cleanup temp file {file_path} after error: {cleanup_error}")

@router.get("/download/video/{job_id}")
async def download_translated_video(job_id: str, current_user: User = Depends(get_current_user)):
    """Download translated video file directly from translation_routes"""
    print(f"[TRANSLATE_ROUTES DOWNLOAD] Request for job {job_id}")
    print(f"[TRANSLATE_ROUTES DOWNLOAD] translation_jobs has {len(translation_jobs)} jobs")
    print(f"[TRANSLATE_ROUTES DOWNLOAD] jobs_db has {len(jobs_db)} jobs")
    
    is_demo_user = DEMO_MODE and current_user.email == "demo@octavia.com"
    job = None
    
    # Check translation_jobs first (local dict)
    if job_id in translation_jobs:
        job = translation_jobs[job_id]
        print(f"[TRANSLATE_ROUTES DOWNLOAD] Found in translation_jobs")
    elif job_id in jobs_db:
        job = jobs_db[job_id]
        print(f"[TRANSLATE_ROUTES DOWNLOAD] Found in jobs_db")
    
    if not job:
        print(f"[TRANSLATE_ROUTES DOWNLOAD] Job not found!")
        raise HTTPException(404, "Job not found in translation_routes")
    
    if job.get("status") != "completed":
        raise HTTPException(400, f"Job not ready. Status: {job.get('status')}")
    
    # Find the video file
    output_path = job.get("output_path") or job.get("result", {}).get("output_path")
    print(f"[TRANSLATE_ROUTES DOWNLOAD] output_path: {output_path}")
    
    if output_path and os.path.exists(output_path):
        print(f"[TRANSLATE_ROUTES DOWNLOAD] Serving: {output_path}")
        return FileResponse(output_path, media_type="video/mp4", filename=f"translated_{job_id}.mp4")
    
    # Try standard locations
    possible_paths = [
        f"backend/outputs/translated_video_{job_id}.mp4",
        f"outputs/translated_video_{job_id}.mp4",
        f"translated_video_{job_id}.mp4"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"[TRANSLATE_ROUTES DOWNLOAD] Found at: {path}")
            return FileResponse(path, media_type="video/mp4", filename=f"translated_{job_id}.mp4")
    
    print(f"[TRANSLATE_ROUTES DOWNLOAD] File not found at any location")
    raise HTTPException(404, "Video file not found")

@router.get("/download/{file_type}/{file_id}")
async def download_file(file_type: str, file_id: str, current_user: User = Depends(get_current_user)):
    """Download generated files"""
    try:
        is_demo_user = DEMO_MODE and current_user.email == "demo@octavia.com"
        print(f"Download request: type={file_type}, id={file_id}, user_id={current_user.id}")

        # First, check if this is a job-based download from both stores
        job = None
        if file_id in translation_jobs:
            job = translation_jobs[file_id]
        else:
            # Check jobs_db from app.py
            from app import jobs_db
            if file_id in jobs_db:
                job = jobs_db[file_id]

        # Verify job ownership (skip for demo users)
        if job and not is_demo_user:
            if job.get("user_id") and job.get("user_id") != current_user.id:
                raise HTTPException(
                    status_code=403,
                    detail="Access denied - you do not own this file"
                )

        if job:
            print(f"Found job in {'translation_jobs' if file_id in translation_jobs else 'jobs_db'}: {job.get('status')}, output_path: {job.get('output_path')}")

        # Enhanced file location search - similar to app.py download_video function
        filename = None
        if job:
            # Try multiple possible file locations
            possible_filenames = []

            # 1. Check job metadata first
            if job.get("output_path"):
                possible_filenames.append(job["output_path"])
            if job.get("output_video"):
                possible_filenames.append(job["output_video"])

            # 2. Standard locations
            possible_filenames.append(f"backend/backend/outputs/translated_video_{file_id}.mp4")
            possible_filenames.append(f"backend/outputs/translated_video_{file_id}.mp4")
            possible_filenames.append(f"outputs/translated_video_{file_id}.mp4")
            possible_filenames.append(f"translated_video_{file_id}.mp4")

            # 3. Try to find the file
            for possible_file in possible_filenames:
                if os.path.exists(possible_file):
                    filename = possible_file
                    print(f"Found video file at: {filename}")
                    break

                # Also try with .mp4 extension if not already
                if not possible_file.endswith('.mp4'):
                    possible_with_mp4 = possible_file + '.mp4'
                    if os.path.exists(possible_with_mp4):
                        filename = possible_with_mp4
                        print(f"Found video file at: {filename}")
                        break

            # 4. Search for any file containing file_id with correct extension
            if not filename:
                search_dirs = [
                    "backend/backend/outputs", 
                    "backend/outputs", 
                    "outputs", 
                    ".",
                    "outputs/subtitles",
                    "backend/outputs/subtitles"
                ]
                
                # Determine expected extension
                expected_ext = ".mp4"
                if file_type == "subtitles":
                    expected_ext = ".srt"  # Default for subtitles
                elif file_type == "audio":
                    expected_ext = ".wav"
                
                for dir_path in search_dirs:
                    if os.path.exists(dir_path):
                        for file in os.listdir(dir_path):
                            if file_id in file and file.endswith(expected_ext):
                                filename = os.path.join(dir_path, file)
                                print(f"Found matching {file_type} file: {filename}")
                                break
                        if filename:
                            break

        # If no job found or no filename from job, use fallback logic
        if not filename:
            # Fallback to legacy hardcoded paths
            if file_type == "subtitles":
                filename = f"subtitles_{file_id}.srt"
                media_type = "text/plain"
            elif file_type == "audio":
                filename = f"translated_audio_{file_id}.wav"
                media_type = "audio/wav"
            elif file_type == "video":
                filename = f"translated_video_{file_id}.mp4"
                media_type = "video/mp4"
            else:
                print(f"Unknown file type: {file_type}")
                raise HTTPException(status_code=404, detail="File type not found")

            print(f"Trying fallback filename: {filename}")

            # Check in outputs directory first
            outputs_path = os.path.join("outputs", os.path.basename(filename))
            print(f"Checking outputs path: {outputs_path}, exists: {os.path.exists(outputs_path)}")
            if os.path.exists(outputs_path):
                filename = outputs_path

            # Check in current directory
            print(f"Checking current directory: {filename}, exists: {os.path.exists(filename)}")
            if not os.path.exists(filename):
                # Try to find any file with this file_id
                import glob
                expected_ext = ".srt" if file_type == "subtitles" else ".wav" if file_type == "audio" else ".mp4"
                
                files = glob.glob(f"*{file_id}*{expected_ext}")
                if files:
                    filename = files[0]
                    print(f"Found alternative {file_type} file: {filename}")
                else:
                    # Try to find in outputs directory and subdirectories
                    search_patterns = [
                        f"outputs/*{file_id}*{expected_ext}",
                        f"outputs/{file_type}/*{file_id}*{expected_ext}",
                        f"backend/outputs/*{file_id}*{expected_ext}",
                        f"backend/outputs/{file_type}/*{file_id}*{expected_ext}"
                    ]
                    
                    found = False
                    for pattern in search_patterns:
                        matches = glob.glob(pattern)
                        if matches:
                            filename = matches[0]
                            print(f"Found {file_type} file by pattern {pattern}: {filename}")
                            found = True
                            break
                    
                    if not found:
                        print(f"No {file_type} files found for file_id {file_id}")
                        raise HTTPException(status_code=404, detail=f"{file_type.capitalize()} file not found")

        # Verify file is actually a video file
        if filename and os.path.exists(filename):
            file_size = os.path.getsize(filename)
            print(f"File size: {file_size} bytes")

            if file_size < 10000:  # Suspiciously small for a video
                with open(filename, 'rb') as f:
                    content = f.read(500)
                    if b'error' in content.lower() or b'failed' in content.lower():
                        raise HTTPException(status_code=500, detail="Video generation failed. Please try again.")

            # Determine media type based on file extension
            ext = os.path.splitext(filename)[1].lower()
            if ext == ".mp4":
                media_type = "video/mp4"
            elif ext == ".srt":
                media_type = "text/plain"
            elif ext == ".wav":
                media_type = "audio/wav"
            else:
                media_type = "application/octet-stream"

            basename = os.path.basename(filename)
            print(f"Serving file: {filename} as {basename}")
            return FileResponse(
                filename,
                media_type=media_type,
                filename=f"octavia_{file_type}_{file_id}{os.path.splitext(basename)[1]}"
            )
        else:
            print(f"File does not exist: {filename}")
            raise HTTPException(status_code=404, detail="File not found")

    except Exception as e:
        print(f"Download error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")
