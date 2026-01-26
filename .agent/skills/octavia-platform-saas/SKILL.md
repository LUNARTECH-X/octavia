# Skill: Octavia Platform SaaS Integration

Expertise in the "Business Layer" of Octavia: Authentication, Payments, and the Credit-based economy.

## Business Logic Components

### 1. Authentication & Users
- **Providers**: Clerk (Frontend) and Supabase (Backend).
- **Implementation**: Managed via `shared_dependencies.py`.
- **Demo Mode**: When `DEMO_MODE=true`, the system bypasses real auth and uses a static `demo@octavia.com` user profile.

### 2. Payment Integration (Polar.sh)
- **Module**: [`payment_routes.py`](file:///c:/Users/onyan/octavia/octavia/backend/routes/payment_routes.py)
- **Features**: Package listings, checkout session creation, and webhook handling.
- **Webhook Logic**: Processes `payment_succeeded` events to automatically inject credits into user accounts.

### 3. Internal Credit System
- **Costs**: 
  - Video Translation: 10 credits.
  - Subtitle Generation: 1 credit.
  - Subtitle Translation: 5 credits.
- **Balance Management**: Credits are tracked in the Supabase `users` table. The `job_service.py` handles deducting credits BEFORE starting a pipeline.

## Core Files
- [`backend/shared_dependencies.py`](file:///c:/Users/onyan/octavia/octavia/backend/shared_dependencies.py): Auth and Supabase initialization.
- [`backend/routes/payment_routes.py`](file:///c:/Users/onyan/octavia/octavia/backend/routes/payment_routes.py): Polar.sh logic.
- [`backend/services/job_service.py`](file:///c:/Users/onyan/octavia/octavia/backend/services/job_service.py): Job initiation and credit checks.

## Maintenance & Testing
- **Test Mode**: Enable `ENABLE_TEST_MODE=true` to simulate successful payments with the 4242... card.
- **Manual Compensation**: If a webhook fails, use `add_user_credits` in `utils.py` to manually adjust a user's balance.
- **Database Schema**: All transactions are logged in the `transactions` table for auditing.
