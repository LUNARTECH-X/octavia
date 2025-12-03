"use client";

import { useRouter } from "next/navigation";
import { XCircle, ArrowLeft, CreditCard } from "lucide-react";
import { motion } from "framer-motion";
import Link from "next/link";

export default function PaymentCancelPage() {
  const router = useRouter();

  return (
    <div className="min-h-screen w-full bg-bg-dark flex items-center justify-center relative overflow-hidden">
      {/* Background glow effects */}
      <div className="glow-purple"
        style={{ width: "600px", height: "600px", position: "absolute", top: "-200px", right: "-100px", zIndex: 0 }} />
      <div className="glow-red"
        style={{ width: "400px", height: "400px", position: "absolute", bottom: "-100px", left: "100px", zIndex: 0 }} />

      <div className="relative z-10 w-full max-w-md p-6">
        <Link href="/" className="inline-flex items-center gap-2 text-slate-400 hover:text-white mb-8 transition-colors">
          <ArrowLeft className="w-4 h-4" />
          Back to Home
        </Link>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="glass-panel p-8 text-center"
        >
          {/* Header */}
          <div className="mb-8">
            <div className="w-16 h-16 mx-auto mb-4">
              <div className="relative w-16 h-16 mx-auto">
                <div className="w-16 h-16 rounded-full bg-red-500/20 flex items-center justify-center border-2 border-red-500">
                  <XCircle className="w-12 h-12 text-red-500" />
                </div>
              </div>
            </div>
            
            <h1 className="text-2xl font-bold text-white mb-2">Payment Cancelled</h1>
            <p className="text-slate-400 text-sm">
              Your payment was cancelled. No charges have been made to your account.
            </p>
          </div>

          {/* Information */}
          <div className="bg-white/5 rounded-xl p-6 mb-8 border border-white/10">
            <div className="flex items-center justify-center gap-3 mb-4">
              <CreditCard className="w-5 h-5 text-slate-400" />
              <span className="text-slate-300 text-sm">No payment was processed</span>
            </div>
            <p className="text-slate-500 text-sm">
              You can return to the billing page to try again or choose a different payment method.
            </p>
          </div>

          {/* Action Buttons */}
          <div className="space-y-3">
            <button
              onClick={() => router.push("/dashboard/billing")}
              className="w-full py-3 rounded-lg bg-primary-purple hover:bg-primary-purple-bright text-white font-semibold transition-colors"
            >
              Return to Billing
            </button>
            
            <Link
              href="/dashboard"
              className="block w-full py-3 rounded-lg bg-white/10 hover:bg-white/20 border border-white/10 text-white transition-colors text-center"
            >
              Go to Dashboard
            </Link>
            
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
              Having issues?{" "}
              <a href="mailto:support@octavia.com" className="text-primary-purple hover:text-white">
                Contact support
              </a>
            </p>
          </div>
        </motion.div>
      </div>
    </div>
  );
}