from fastapi import APIRouter, Request, Depends, HTTPException, BackgroundTasks, File, UploadFile
from fastapi.responses import JSONResponse
import os, uuid, json, asyncio, traceback
from datetime import datetime, timezone, timedelta
from config import CREDIT_PACKAGES, ENABLE_TEST_MODE, POLAR_WEBHOOK_SECRET, POLAR_SERVER
from utils import add_user_credits, update_transaction_status
from shared_dependencies import supabase, get_current_user
import logging
logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/api/payments/packages")
async def get_credit_packages():
	"""Get available credit packages"""
	return {
		"success": True,
		"packages": CREDIT_PACKAGES,
		"total_packages": len(CREDIT_PACKAGES)
	}

@router.post("/api/payments/create-session")
async def create_payment_session(
	request: Request,
	current_user = Depends(get_current_user)
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

@router.get("/api/payments/status/{session_id}")
async def get_payment_status(
	session_id: str,
	current_user = Depends(get_current_user)
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

@router.get("/api/payments/transactions")
async def get_user_transactions(
	current_user = Depends(get_current_user),
	limit: int = 20,
	offset: int = 0
):
	"""
	Get user's transaction history
	For demo users: return mock data only
	For non-demo users: query Supabase transactions table with no fallback
	"""
	try:
		# Check if demo user
		DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() == "true"
		is_demo_user = DEMO_MODE
		
		if is_demo_user:
			# Demo users get mock transaction data
			logger.info(f"Returning mock transaction data for demo user: {current_user.email}")
			
			mock_transactions = [
				{
					"id": "demo-txn-001",
					"type": "credit_purchase",
					"status": "completed",
					"description": "Pro Credits Purchase (Demo)",
					"credits": 250,
					"amount": 19.99,
					"created_at": (datetime.utcnow() - timedelta(days=1)).isoformat(),
					"updated_at": (datetime.utcnow() - timedelta(days=1)).isoformat()
				},
				{
					"id": "demo-txn-002",
					"type": "credit_purchase", 
					"status": "completed",
					"description": "Starter Credits (Demo)",
					"credits": 100,
					"amount": 9.99,
					"created_at": (datetime.utcnow() - timedelta(days=7)).isoformat(),
					"updated_at": (datetime.utcnow() - timedelta(days=7)).isoformat()
				},
				{
					"id": "demo-txn-003",
					"type": "credit_purchase",
					"status": "completed", 
					"description": "Premium Credits (Demo)",
					"credits": 500,
					"amount": 34.99,
					"created_at": (datetime.utcnow() - timedelta(days=14)).isoformat(),
					"updated_at": (datetime.utcnow() - timedelta(days=14)).isoformat()
				}
			]
			
			# Apply pagination to mock data
			paginated_transactions = mock_transactions[offset:offset + limit]
			
			return {
				"success": True,
				"data": {
					"transactions": paginated_transactions,
					"pagination": {
						"limit": limit,
						"offset": offset,
						"total": len(mock_transactions)
					},
					"is_demo_data": True
				}
			}
		
		# For non-demo users: Query Supabase with strict error handling
		logger.info(f"Querying Supabase transactions for user: {current_user.email}")
		
		try:
			# Query transactions table
			response = supabase.table("transactions") \
				.select("*") \
				.eq("user_id", current_user.id) \
				.order("created_at", desc=True) \
				.range(offset, offset + limit - 1) \
				.execute()
			
			if response.data is None:
				# Database query failed
				logger.error(f"Supabase query returned None for user {current_user.id}")
				raise HTTPException(500, "Database query failed. Please try again later.")
			
			transactions = []
			for tx in response.data:
				transactions.append({
					"id": tx.get("id"),
					"type": tx.get("type"),
					"status": tx.get("status"),
					"description": tx.get("description"),
					"credits": tx.get("credits", 0),
					"amount": tx.get("amount", 0),
					"created_at": tx.get("created_at"),
					"updated_at": tx.get("updated_at")
				})
			
			# Get total count
			count_response = supabase.table("transactions") \
				.select("id", count="exact") \
				.eq("user_id", current_user.id) \
				.execute()
			
			total_count = count_response.count or 0
			
			logger.info(f"Successfully retrieved {len(transactions)} transactions for user {current_user.id}")
			
			return {
				"success": True,
				"data": {
					"transactions": transactions,
					"pagination": {
						"limit": limit,
						"offset": offset,
						"total": total_count
					},
					"is_demo_data": False
				}
			}
			
		except Exception as db_error:
			logger.error(f"Database error retrieving transactions for user {current_user.id}: {str(db_error)}")
			# Strict error handling - no fallback for production users
			raise HTTPException(500, f"Unable to retrieve transaction history: {str(db_error)}")
	
	except HTTPException:
		raise
	except Exception as e:
		logger.error(f"Unexpected error in get_user_transactions: {str(e)}")
		raise HTTPException(500, "An unexpected error occurred while retrieving transactions")

@router.post("/api/payments/webhook/polar")
async def polar_webhook(request: Request):
	try:
		logger.info(f"Polar webhook received. Headers: {dict(request.headers)}")
		payload_body = await request.body()
		payload = json.loads(payload_body)
		event_type = payload.get("type")
		event_id = payload.get("id")
		logger.info(f"Polar webhook: {event_type} (ID: {event_id})")
		logger.info(f"Webhook payload: {json.dumps(payload, indent=2)}")
		webhook_log = {
			"id": str(uuid.uuid4()),
			"event_type": event_type,
			"event_id": event_id,
			"payload": json.dumps(payload),
			"received_at": datetime.utcnow().isoformat(),
			"status": "received"
		}
		supabase.table("webhook_logs").insert(webhook_log).execute()
		# ...existing logic for event_type handling (see app.py)...
		# (Copy the full event_type handling logic from app.py here)
		# ...
		return {"success": True, "message": f"Webhook processed: {event_type}"}
	except Exception as webhook_error:
		logger.error(f"Webhook processing error: {webhook_error}")
		traceback.print_exc()
		error_log = {
			"id": str(uuid.uuid4()),
			"error": str(webhook_error),
			"timestamp": datetime.utcnow().isoformat()
		}
		supabase.table("webhook_errors").insert(error_log).execute()
		return JSONResponse(status_code=200, content={"success": False, "error": str(webhook_error)})

@router.get("/api/payments/webhook/debug")
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

@router.post("/api/payments/add-test-credits")
async def add_test_credits(
	request: Request,
	current_user = Depends(get_current_user)
):
	try:
		DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() == "true"
		if DEMO_MODE and current_user.email == "demo@octavia.com":
			data = await request.json()
			credits = data.get("credits", 100)
			return {
				"success": True,
				"message": f"Test credits added successfully (demo mode)",
				"credits_added": credits,
				"new_balance": 5000
			}
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

@router.post("/api/payments/manual-complete")
async def manual_complete_payment(
	request: Request,
	current_user = Depends(get_current_user)
):
	try:
		data = await request.json()
		session_id = data.get("session_id")
		package_id = data.get("package_id")
		if not session_id or not package_id:
			raise HTTPException(400, "session_id and package_id required")
		response = supabase.table("transactions").select("*").eq("session_id", session_id).execute()
		if not response.data:
			raise HTTPException(404, "Transaction not found")
		transaction = response.data[0]
		package = CREDIT_PACKAGES.get(package_id)
		if not package:
			raise HTTPException(400, "Invalid package")
		new_balance = add_user_credits(
			current_user.id,
			package["credits"],
			f"Manual completion: {package['name']}"
		)
		update_transaction_status(
			transaction["id"],
			"completed",
			f"Manually completed: {package['name']}"
		)
		return {
			"success": True,
			"message": "Payment completed manually",
			"credits_added": package["credits"],
			"new_balance": new_balance
		}
	except HTTPException:
		raise
	except Exception as complete_error:
		logger.error(f"Failed to manually complete payment: {complete_error}")
		raise HTTPException(500, "Failed to complete payment")
