"use client";

import { useEffect, Suspense } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { Loader2 } from "lucide-react";

function AuthCallbackContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const token = searchParams.get('token');
  const email = searchParams.get('email');

  // Handle OAuth or email verification callbacks
  useEffect(() => {
    if (token && email) {
      // Store authentication token in localStorage
      localStorage.setItem('octavia_user', JSON.stringify({
        email,
        token,
        // Default values - actual user data should be fetched from API
        name: 'User',
        credits: 1000,
        verified: true
      }));
      
      // Redirect to dashboard after successful authentication
      router.push('/dashboard');
    } else {
      // Missing required parameters, redirect to login
      router.push('/login');
    }
  }, [token, email, router]);

  return (
    <div className="min-h-screen w-full bg-bg-dark flex items-center justify-center">
      <div className="text-center">
        <Loader2 className="w-12 h-12 text-primary-purple animate-spin mx-auto mb-4" />
        <p className="text-white text-lg">Completing authentication...</p>
        <p className="text-slate-400 text-sm mt-2">Redirecting to dashboard</p>
      </div>
    </div>
  );
}

export default function AuthCallbackPage() {
  return (
    <Suspense fallback={
      <div className="min-h-screen w-full bg-bg-dark flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="w-12 h-12 text-primary-purple animate-spin mx-auto mb-4" />
          <p className="text-white text-lg">Loading...</p>
        </div>
      </div>
    }>
      <AuthCallbackContent />
    </Suspense>
  );
}