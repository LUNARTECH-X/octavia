"""
Unified job storage service using Supabase for persistence
"""
import os
import asyncio
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from shared_dependencies import supabase


class JobStorage:
    """Unified job storage using Supabase for regular users, local storage for demo users"""

    def __init__(self):
        self.table_name = "translation_jobs"
        self.demo_file = "demo_jobs.json"
        self._demo_jobs = self._load_demo_jobs()

    def _load_demo_jobs(self) -> Dict:
        """Load demo jobs from file"""
        import json
        if os.path.exists(self.demo_file):
            try:
                with open(self.demo_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading demo jobs: {e}")
        return {}

    def _save_demo_jobs(self):
        """Save demo jobs to file"""
        import json
        try:
            with open(self.demo_file, "w") as f:
                json.dump(self._demo_jobs, f)
        except Exception as e:
            print(f"Error saving demo jobs: {e}")

    def _is_demo_user(self, user_id: str = None, job_data: dict = None) -> bool:
        """Check if this is a demo user"""
        # Demo user IDs (from database)
        DEMO_USER_IDS = [
            "550e8400-e29b-41d4-a716-446655440000",  # Legacy demo ID
            "00affbea-0235-4cac-a1c5-e74b89342dd3",  # Current demo ID
            "demo-user-id",                          # Generic frontend demo ID
        ]
        
        # Check environment variable
        if os.getenv("DEMO_MODE") == "true":
            return True
            
        # Check user_id if provided
        if user_id and user_id in DEMO_USER_IDS:
            return True

        # Check job_data for demo indicators
        if job_data:
            if job_data.get("user_email") == "demo@octavia.com":
                return True
            if job_data.get("user_id") in DEMO_USER_IDS:
                return True

        return False
    
    async def create_job(self, job_data: dict) -> str:
        """Create a new job - uses local storage for demo users, Supabase for others"""
        try:
            job_id = job_data.get("id")
            if not job_id:
                raise ValueError("Job ID is required")

            # Set default values
            defaults = {
                "status": "pending",
                "progress": 0,
                "result": {},
                "metrics": {},
                "created_at": datetime.utcnow().isoformat(),
                "version": 0
            }

            for key, value in defaults.items():
                if key not in job_data:
                    job_data[key] = value

            # Check if this is a demo user
            if self._is_demo_user(job_data=job_data):
                # Store in local memory and file for demo users
                self._demo_jobs[job_id] = job_data.copy()
                self._save_demo_jobs()
                print(f"Created demo job {job_id} in persistent local storage")
                return job_id
            else:
                # Store in Supabase for regular users
                from services.db_utils import with_retry
                
                async def insert_job():
                    return supabase.table(self.table_name).insert(job_data).execute()
                
                result = await with_retry(insert_job)

                if result.data:
                    print(f"Created job {job_id} in Supabase")
                    return job_id
                else:
                    raise Exception("Failed to create job")
        except Exception as e:
            print(f"Error creating job: {e}")
            raise
    
    async def get_job(self, job_id: str) -> Optional[Dict]:
        """Get a job by ID - checks demo storage first, then Supabase"""
        try:
            # Check demo jobs first
            if job_id in self._demo_jobs:
                print(f"Found demo job {job_id} in local storage")
                return self._demo_jobs[job_id].copy()

            # Check Supabase for regular jobs
            from services.db_utils import with_retry
            
            async def fetch_job():
                return supabase.table(self.table_name).select("*").eq("id", job_id).execute()
                
            result = await with_retry(fetch_job)
            if result.data:
                return result.data[0]
            return None
        except Exception as e:
            print(f"Error getting job {job_id}: {e}")
            return None
    
    async def get_user_jobs(self, user_id: str) -> List[Dict]:
        """Get all jobs for a user - includes demo jobs if user is demo"""
        try:
            jobs = []

            # Check if this is a demo user
            if self._is_demo_user(user_id=user_id):
                # Get demo jobs
                demo_jobs = [job for job in self._demo_jobs.values() if job.get("user_id") == user_id]
                jobs.extend(demo_jobs)
                print(f"Found {len(demo_jobs)} demo jobs for user {user_id}")
            else:
                # Get regular jobs from Supabase
                from services.db_utils import with_retry
                
                async def fetch_user_jobs():
                    return supabase.table(self.table_name).select("*").eq("user_id", user_id).order("created_at", desc=True).execute()
                
                result = await with_retry(fetch_user_jobs)
                jobs.extend(result.data if result.data else [])

            return jobs
        except Exception as e:
            print(f"Error getting jobs for user {user_id}: {e}")
            return []
    
    async def update_job(self, job_id: str, updates: dict) -> bool:
        """Update job fields - uses local storage for demo jobs, Supabase for others"""
        try:
            job = await self.get_job(job_id)
            if not job:
                return False

            # Check if this is a demo job
            if job_id in self._demo_jobs:
                # Update in local storage
                self._demo_jobs[job_id].update(updates)
                self._demo_jobs[job_id]["updated_at"] = datetime.utcnow().isoformat()
                self._save_demo_jobs()
                print(f"Updated demo job {job_id} in persistent local storage")
                return True
            else:
                # Update in Supabase for regular jobs
                from services.db_utils import with_retry
                
                current_version = job.get("version", 0)
                updates["version"] = current_version + 1

                async def perform_update():
                    return supabase.table(self.table_name).update(updates).eq("id", job_id).eq("version", current_version).execute()
                
                result = await with_retry(perform_update)

                return len(result.data) > 0
        except Exception as e:
            print(f"Error updating job {job_id}: {e}")
            return False
    
    async def update_progress(self, job_id: str, progress: int, message: str = "") -> bool:
        """Update job progress"""
        updates = {
            "progress": progress,
            "message": message
        }
        return await self.update_job(job_id, updates)
    
    async def update_status(self, job_id: str, status: str) -> bool:
        """Update job status"""
        updates = {"status": status}
        if status == "completed":
            updates["completed_at"] = datetime.utcnow().isoformat()
        elif status == "failed":
            updates["failed_at"] = datetime.utcnow().isoformat()
        
        return await self.update_job(job_id, updates)
    
    async def update_eta(self, job_id: str, eta_seconds: float) -> bool:
        """Update ETA for job"""
        return await self.update_job(job_id, {"eta_seconds": eta_seconds})
    
    async def update_metrics(self, job_id: str, metrics: dict) -> bool:
        """Update job metrics"""
        return await self.update_job(job_id, {"metrics": metrics})
    
    async def complete_job(self, job_id: str, result: dict, output_path: str = None) -> bool:
        """Mark job as completed with results - merges result fields directly into job"""
        updates = {
            "status": "completed",
            "progress": 100,
            "completed_at": datetime.utcnow().isoformat()
        }
        
        # Merge all result fields directly into the job (not nested)
        # This ensures fields like 'filename', 'download_url', 'segment_count' are accessible
        for key, value in result.items():
            updates[key] = value
        
        if output_path:
            updates["output_path"] = output_path
        
        return await self.update_job(job_id, updates)
    
    async def fail_job(self, job_id: str, error: str) -> bool:
        """Mark job as failed with error message"""
        updates = {
            "status": "failed",
            "error": error,
            "failed_at": datetime.utcnow().isoformat()
        }
        return await self.update_job(job_id, updates)

    async def cancel_job(self, job_id: str) -> bool:
        """Mark job as cancelled and set cancellation flag"""
        updates = {
            "status": "cancelled",
            "cancelled": True,
            "cancelled_at": datetime.utcnow().isoformat(),
            "message": "Job cancelled by user"
        }
        return await self.update_job(job_id, updates)

    async def is_cancelled(self, job_id: str) -> bool:
        """Check if a job has been cancelled"""
        try:
            job = await self.get_job(job_id)
            if job:
                return job.get("cancelled", False) or job.get("status") == "cancelled"
            return False
        except Exception as e:
            print(f"Error checking cancellation for job {job_id}: {e}")
            return False


# Global instance
job_storage = JobStorage()

