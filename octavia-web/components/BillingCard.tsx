"use client";

import { useState } from "react";
import { Check, CreditCard, Loader2 } from "lucide-react";
import { api } from "@/lib/api";
import { useUser } from "@/contexts/UserContext";

interface BillingCardProps {
  package: {
    id: string;
    name: string;
    credits: number;
    price: number;
    description: string;
    features: string[];
    popular?: boolean;
    icon?: React.ReactNode;
  };
}

export default function BillingCard({ package: pkg }: BillingCardProps) {
  const { user, fetchUserProfile } = useUser();
  const [isProcessing, setIsProcessing] = useState(false);
  const [statusMessage, setStatusMessage] = useState("");

  const handlePurchase = async () => {
    if (!user) return;

    setIsProcessing(true);
    setStatusMessage("Creating payment session...");

    try {
      // Create payment session
      const response = await api.createPaymentSession(pkg.id);

      if (response.success && response.data?.checkout_url) {
        setStatusMessage("Redirecting to payment...");

        // Store session info for polling
        if (response.data?.session_id) {
          localStorage.setItem('last_payment_session', JSON.stringify({
            session_id: response.data.session_id,
            transaction_id: response.data.transaction_id,
            package_id: pkg.id,
            timestamp: Date.now()
          }));
        }

        // For test mode, handle directly
        if (response.data?.test_mode) {
          setStatusMessage("Adding test credits...");

          // Poll for completion
          await pollPaymentStatus(response.data.session_id, response.data.transaction_id);
        } else {
          // Redirect to Polar.sh checkout
          window.location.href = response.data.checkout_url;
        }
      } else {
        setStatusMessage(response.error || "Failed to create payment session");
      }
    } catch (error) {
      console.error("Purchase error:", error);
      setStatusMessage("Payment failed. Please try again.");
    } finally {
      setIsProcessing(false);
    }
  };

  const pollPaymentStatus = async (sessionId: string, transactionId: string) => {
    const maxAttempts = 10;
    let attempts = 0;

    const userStr = localStorage.getItem('octavia_user');
    const userData = userStr ? JSON.parse(userStr) : null;
    const token = userData?.token;

    while (attempts < maxAttempts) {
      try {
        const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/payments/status/${sessionId}`, {
          headers: token ? {
            'Authorization': `Bearer ${token}`
          } : {},
        });

        if (response.ok) {
          const data = await response.json();

          if (data.status === 'completed') {
            setStatusMessage("Payment completed! Updating credits...");
            await fetchUserProfile(); // Refresh user data
            localStorage.removeItem('last_payment_session');
            setTimeout(() => setStatusMessage(""), 3000);
            return;
          } else if (data.status === 'failed') {
            setStatusMessage("Payment failed. Please try again.");
            return;
          }
          // Still pending, continue polling
        }
      } catch (error) {
        console.error("Polling error:", error);
      }

      attempts++;
      await new Promise(resolve => setTimeout(resolve, 1000)); // Wait 1 second
    }

    setStatusMessage("Payment status check timed out. Please refresh page to check manually.");
  };

  // Check for pending payments on component mount
  const checkPendingPayments = async () => {
    const pendingPayment = localStorage.getItem('last_payment_session');
    if (pendingPayment) {
      const paymentData = JSON.parse(pendingPayment);
      const timeElapsed = Date.now() - paymentData.timestamp;

      // Only check if it was less than 5 minutes ago
      if (timeElapsed < 5 * 60 * 1000) {
        setIsProcessing(true);
        setStatusMessage("Checking payment status...");
        await pollPaymentStatus(paymentData.session_id, paymentData.transaction_id);
        setIsProcessing(false);
      } else {
        localStorage.removeItem('last_payment_session');
      }
    }
  };

  // Run check on mount
  useState(() => {
    checkPendingPayments();
  });

  return (
    <div className={`glass-panel p-6 relative flex flex-col h-full ${pkg.popular ? 'border-2 border-primary-purple' : 'border border-white/10'}`}>
      {pkg.popular && (
        <div className="absolute -top-3 left-1/2 transform -translate-x-1/2 bg-gradient-to-r from-primary-purple to-accent-cyan text-white text-[11px] font-bold px-4 py-1 rounded-full z-20 shadow-[0_0_15px_rgba(168,85,247,0.5)] whitespace-nowrap">
          Most Popular
        </div>
      )}

      <div className="text-center mb-6">
        <div className="inline-flex items-center justify-center w-12 h-12 rounded-full bg-primary-purple/20 text-primary-purple mb-4">
          {pkg.icon || <CreditCard className="w-6 h-6" />}
        </div>
        <h3 className="text-xl font-bold text-white mb-2">{pkg.name}</h3>
        <p className="text-slate-400 text-sm mb-4">{pkg.description}</p>

        <div className="mb-4">
          <div className="text-4xl font-black text-white">${pkg.price}</div>
          <div className="text-slate-400 text-sm mt-1">{pkg.credits.toLocaleString()} Credits</div>
        </div>
      </div>

      <ul className="space-y-3 mb-8 flex-grow">
        {pkg.features.map((feature, idx) => (
          <li key={idx} className="flex items-center gap-3 text-sm">
            <Check className="w-4 h-4 text-accent-green flex-shrink-0" />
            <span className="text-slate-300">{feature}</span>
          </li>
        ))}
      </ul>

      <button
        onClick={handlePurchase}
        disabled={isProcessing}
        className={`w-full py-3 rounded-lg font-bold transition-all duration-200 ${pkg.popular
          ? 'bg-gradient-to-r from-primary-purple to-accent-cyan hover:opacity-90 text-white'
          : 'bg-white/10 hover:bg-white/20 text-white'
          } ${isProcessing ? 'opacity-70 cursor-not-allowed' : ''}`}
      >
        {isProcessing ? (
          <span className="flex items-center justify-center gap-2">
            <Loader2 className="w-4 h-4 animate-spin" />
            Processing...
          </span>
        ) : (
          `Buy ${pkg.name}`
        )}
      </button>

      {statusMessage && (
        <div className={`mt-4 text-sm text-center p-3 rounded-lg ${statusMessage.includes("completed") || statusMessage.includes("success")
          ? "bg-accent-cyan/10 text-accent-cyan border border-accent-cyan/20"
          : statusMessage.includes("failed") || statusMessage.includes("error")
            ? "bg-red-500/10 text-red-400 border border-red-500/20"
            : "bg-blue-500/10 text-blue-400 border border-blue-500/20"
          }`}>
          {statusMessage}
        </div>
      )}
    </div>
  );
}