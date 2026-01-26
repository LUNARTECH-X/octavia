# Skill: Octavia Frontend Mastery

Expertise in the Octavia web dashboard, built with Next.js 18 (App Router), shadcn/ui, and Tailwind CSS.

## Frontend Architecture

The frontend is located in [`octavia-web/`](file:///c:/Users/onyan/octavia/octavia/octavia-web). It is designed for real-time interaction with the translation backend.

### Core Technologies
- **Next.js 18**: App Router for server-side rendering and routing.
- **shadcn/ui + Lucide**: For the premium, dark-themed UI components.
- **Framer Motion**: For smooth micro-animations and transitions.
- **TanStack Query (React Query)**: For state management and API data fetching.

### High-Impact Components
- **Dashboard**: The central hub for managing video/audio/subtitle jobs.
- **SSE Progress Tracking**: Custom hooks used to listen to the backend's Server-Sent Events for real-time job status.
- **Magic Toggle Group**: UI controls for enabling features like "Magic Mode" and "Voice Selection."

## Integration Patterns
- **API Communication**: All requests are routed through `NEXT_PUBLIC_API_URL` (usually :8000).
- **Authentication**: Integrated with Clerk for user/org management and Supabase for session persistence.
- **Payments**: Uses Polar.sh checkout components for credit purchases.

## Common UI Tasks
- **Updating the Theme**: Modifications should be made in `globals.css` using CSS variables for consistency.
- **Adding a New Page**: Follow the App Router pattern (`app/[page-name]/page.tsx`).
- **Fixing Dashboard Glitches**: Check the `contexts` and `hooks` directories for state management issues.

## Debugging Frontend Issues
- **CORS Errors**: Verify the `CORS_ORIGINS` in the backend `.env` matches the frontend URL.
- **Hydration Errors**: Ensure that the `lucide-react` icons are used correctly inside Client Components.
- **SSE Dropouts**: Check the browser network tab for persistent connection failures to `/api/translate/status`.
