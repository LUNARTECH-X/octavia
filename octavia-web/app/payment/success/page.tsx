"use client";

import { useEffect, useState, Suspense } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { CheckCircle, ArrowRight, Loader2, PartyPopper } from "lucide-react";
import { motion } from "framer-motion";
import Link from "next/link";
import { api } from "@/lib/api";
import { useUser } from "@/contexts/UserContext";

function PaymentSuccessContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const { user, fetchUserProfile } = useUser();
  const [loading, setLoading] = useState(true);
  const [success, setSuccess] = useState(false);
  const [message, setMessage] = useState("");

  const sessionId = searchParams.get("session_id");
  const packageId = searchParams.get("package_id");

  useEffect(() => {
    if (!sessionId || !packageId) {
      // No payment session info, redirect to billing
      router.push("/dashboard/billing");
      return;
    }

    // In a real app, you would verify the payment with your backend
    // For now, we'll simulate a successful payment
    const verifyPayment = async () => {
      setLoading(true);
      
      try {
        // Simulate API call to verify payment
        await new Promise(resolve => setTimeout(resolve, 1500));
        
        // Fetch updated user profile to get new credits
        await fetchUserProfile();
        
        setSuccess(true);
        setMessage("Payment successful! Credits have been added to your account.");
        
      } catch (error) {
        setSuccess(false);
        setMessage("Payment verification failed. Please contact support.");
        console.error("Payment verification error:", error);
      } finally {
        setLoading(false);
      }
    };

    verifyPayment();
  }, [sessionId, packageId, router, fetchUserProfile]);

  return (
    <div className="min-h-screen w-full bg-bg-dark flex items-center justify-center relative overflow-hidden">
      {/* Background glow effects */}
      <div className="glow-purple-strong"
        style={{ width: "600px", height: "600px", position: "absolute", top: "-200px", right: "-100px", zIndex: 0 }} />
      <div className="glow-green"
        style={{ width: "400px", height: "400px", position: "absolute", bottom: "-100px", left: "100px", zIndex: 0 }} />

      <div className="relative z-10 w-full max-w-md p-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="glass-panel p-8 text-center"
        >
          {/* Header */}
          <div className="mb-8">
            <div className="w-16 h-16 mx-auto mb-4 relative">
              {loading ? (
                <Loader2 className="w-16 h-16 text-primary-purple animate-spin" />
              ) : success ? (
                <div className="relative">
                  <CheckCircle className="w-16 h-16 text-green-500" />
                  <PartyPopper className="w-8 h-8 text-yellow-500 absolute -top-2 -right-2" />
                </div>
              ) : (
                <div className="w-16 h-16 rounded-full bg-red-500/20 flex items-center justify-center border-2 border-red-500">
                  <div className="text-red-500 text-3xl">!</div>
                </div>
              )}
            </div>
            
            <h1 className="text-2xl font-bold text-white mb-2">
              {loading ? "Verifying Payment..." : 
               success ? "Payment Successful!" : "Payment Failed"}
            </h1>
            <p className="text-slate-400 text-sm">
              {loading ? "Please wait while we confirm your payment..." : message}
            </p>
          </div>

          {/* Payment Details */}
          {!loading && success && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
              className="bg-white/5 rounded-xl p-6 mb-8 border border-white/10"
            >
              <div className="grid grid-cols-2 gap-4">
                <div className="text-left">
                  <div className="text-slate-500 text-xs uppercase tracking-wider mb-1">Order ID</div>
                  <div className="text-white font-medium text-sm truncate">{sessionId}</div>
                </div>
                <div className="text-right">
                  <div className="text-slate-500 text-xs uppercase tracking-wider mb-1">Package</div>
                  <div className="text-white font-medium text-sm">{packageId?.replace("_", " ")}</div>
                </div>
                <div className="text-left">
                  <div className="text-slate-500 text-xs uppercase tracking-wider mb-1">Status</div>
                  <div className="text-green-400 font-medium text-sm">Completed</div>
                </div>
                <div className="text-right">
                  <div className="text-slate-500 text-xs uppercase tracking-wider mb-1">New Balance</div>
                  <div className="text-2xl font-bold text-white">{user?.credits?.toLocaleString()}</div>
                </div>
              </div>
            </motion.div>
          )}

          {/* Action Buttons */}
          <div className="space-y-3">
            {success ? (
              <>
                <Link
                  href="/dashboard"
                  className="block w-full py-3 rounded-lg bg-gradient-to-r from-primary-purple to-primary-purple-bright text-white font-semibold hover:opacity-90 transition-opacity flex items-center justify-center gap-2"
                >
                  Go to Dashboard
                  <ArrowRight className="w-4 h-4" />
                </Link>
                <Link
                  href="/dashboard/billing"
                  className="block w-full py-3 rounded-lg bg-white/10 hover:bg-white/20 border border-white/10 text-white transition-colors"
                >
                  View Billing
                </Link>
              </>
            ) : !loading ? (
              <>
                <button
                  onClick={() => router.push("/dashboard/billing")}
                  className="w-full py-3 rounded-lg bg-primary-purple hover:bg-primary-purple-bright text-white font-semibold transition-colors"
                >
                  Try Again
                </button>
                <Link
                  href="/support"
                  className="block w-full py-3 rounded-lg bg-white/10 hover:bg-white/20 border border-white/10 text-white transition-colors text-center"
                >
                  Contact Support
                </Link>
              </>
            ) : null}
            
            <Link
              href="/"
              className="block w-full py-3 text-center text-slate-400 hover:text-white transition-colors text-sm"
            >
              Back to Home
            </Link>
          </div>

          {/* Support Information */}
          <div className="pt-6 mt-6 border-t border-white/10">
            <p className="text-sm text-slate-500">
              Need help?{" "}
              <a href="mailto:support@octavia.com" className="text-primary-purple hover:text-white">
                support@octavia.com
              </a>
            </p>
          </div>
        </motion.div>
      </div>
    </div>
  );
}

export default function PaymentSuccessPage() {
  return (
    <Suspense fallback={
      <div className="min-h-screen w-full bg-bg-dark flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="w-12 h-12 text-primary-purple animate-spin mx-auto mb-4" />
          <p className="text-white text-lg">Loading payment details...</p>
        </div>
      </div>
    }>
      <PaymentSuccessContent />
    </Suspense>
  );
}