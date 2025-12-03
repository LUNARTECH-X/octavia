"use client";

import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { CreditCard, Check, Shield, Zap, DollarSign, RefreshCw, AlertCircle } from "lucide-react";
import BillingCard from "@/components/BillingCard";
import { api } from "@/lib/api";
import { useUser } from "@/contexts/UserContext";

// Define credit packages
const creditPackages = [
  {
    id: "starter_credits",
    name: "Starter",
    credits: 100,
    price: 9.99,
    description: "Perfect for getting started",
    features: [
      "100 translation credits",
      "Standard processing",
      "Email support",
      "7-day credit validity"
    ],
    icon: <DollarSign className="w-6 h-6" />
  },
  {
    id: "pro_credits",
    name: "Pro",
    credits: 250,
    price: 19.99,
    description: "Best for regular users",
    features: [
      "250 translation credits",
      "Priority processing",
      "Priority email support",
      "30-day credit validity",
      "Video preview included"
    ],
    popular: true,
    icon: <Zap className="w-6 h-6" />
  },
  {
    id: "premium_credits",
    name: "Premium",
    credits: 500,
    price: 34.99,
    description: "For power users & businesses",
    features: [
      "500 translation credits",
      "Express processing",
      "24/7 priority support",
      "60-day credit validity",
      "All features included",
      "Batch upload support"
    ],
    icon: <Shield className="w-6 h-6" />
  }
];

interface Transaction {
  id: string;
  amount: number;
  credits: number;
  status: string;
  created_at: string;
  description: string;
}

export default function BillingPage() {
  const { user, fetchUserProfile } = useUser();
  const [transactions, setTransactions] = useState<Transaction[]>([]);
  const [loadingTransactions, setLoadingTransactions] = useState(false);
  const [refreshingCredits, setRefreshingCredits] = useState(false);
  const [showSuccess, setShowSuccess] = useState(false);
  const [successMessage, setSuccessMessage] = useState("");

  // Fetch transaction history and check for completed payments
  useEffect(() => {
    fetchTransactions();
    checkForCompletedPayments();
  }, []);

  // Check if we just returned from a payment
  const checkForCompletedPayments = async () => {
    const urlParams = new URLSearchParams(window.location.search);
    const paymentSuccess = urlParams.get('payment_success');
    const sessionId = urlParams.get('session_id');
    
    if (paymentSuccess === 'true' && sessionId) {
      setShowSuccess(true);
      setSuccessMessage("Payment completed! Updating your credits...");
      
      // Poll for payment completion
      await pollPaymentStatus(sessionId);
      
      // Clean up URL
      window.history.replaceState({}, document.title, window.location.pathname);
    }
    
    // Check localStorage for pending payments
    const pendingPayment = localStorage.getItem('last_payment_session');
    if (pendingPayment) {
      const paymentData = JSON.parse(pendingPayment);
      await pollPaymentStatus(paymentData.session_id);
    }
  };

  const pollPaymentStatus = async (sessionId: string) => {
    const userStr = localStorage.getItem('octavia_user');
    const userData = userStr ? JSON.parse(userStr) : null;
    const token = userData?.token;
    
    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/payments/status/${sessionId}`, {
        headers: token ? {
          'Authorization': `Bearer ${token}`
        } : {},
      });
      
      if (response.ok) {
        const data = await response.json();
        
        if (data.status === 'completed') {
          setSuccessMessage(`Success! Added ${data.credits} credits to your account.`);
          await fetchUserProfile(); // Refresh credits
          await fetchTransactions(); // Refresh transactions
          localStorage.removeItem('last_payment_session');
        } else if (data.status === 'failed') {
          setSuccessMessage("Payment failed. Please try again.");
          setShowSuccess(true);
        }
      }
    } catch (error) {
      console.error("Error checking payment status:", error);
    }
  };

  const fetchTransactions = async () => {
    setLoadingTransactions(true);
    try {
      const response = await api.getTransactionHistory();
      if (response.success && response.transactions) {
        setTransactions(response.transactions);
      }
    } catch (error) {
      console.error("Failed to fetch transactions:", error);
    } finally {
      setLoadingTransactions(false);
    }
  };

  const refreshCredits = async () => {
    setRefreshingCredits(true);
    try {
      await fetchUserProfile();
    } finally {
      setRefreshingCredits(false);
    }
  };

  return (
    <div className="space-y-8">
      {/* Success Notification */}
      {showSuccess && (
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="glass-panel p-4 border border-green-500/30 bg-green-500/10"
        >
          <div className="flex items-center gap-3">
            <Check className="w-5 h-5 text-green-400 flex-shrink-0" />
            <div className="flex-1">
              <p className="text-green-300 font-medium">{successMessage}</p>
            </div>
            <button
              onClick={() => setShowSuccess(false)}
              className="text-green-400 hover:text-green-300"
            >
              ×
            </button>
          </div>
        </motion.div>
      )}

      {/* Header */}
      <div>
        <h1 className="font-display text-3xl font-black text-white mb-2 text-glow-purple">Plans & Billing</h1>
        <p className="text-slate-400 text-sm">Manage your subscription and payment methods</p>
      </div>

      {/* Credit Balance Section */}
      <div className="glass-panel p-8 relative overflow-hidden">
        <div className="glass-shine" />
        <div className="relative z-10">
          <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-6 mb-8">
            <div>
              <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-primary-purple/20 border border-primary-purple/30 text-primary-purple-bright text-xs font-bold uppercase tracking-wider mb-4">
                Credit Balance
              </div>
              <div className="flex items-center gap-4">
                <h2 className="text-3xl font-bold text-white">
                  {user?.credits?.toLocaleString() || 0} Credits
                </h2>
                <button
                  onClick={refreshCredits}
                  disabled={refreshingCredits}
                  className="flex items-center gap-2 px-3 py-1 rounded-lg bg-white/10 hover:bg-white/20 transition-colors text-sm text-slate-300"
                >
                  <RefreshCw className={`w-4 h-4 ${refreshingCredits ? 'animate-spin' : ''}`} />
                  Refresh
                </button>
              </div>
              <p className="text-slate-400 max-w-md mt-2">
                Credits are used for video translations. Each minute of video uses approximately 5 credits.
                {process.env.NEXT_PUBLIC_ENABLE_TEST_PAYMENTS === "true" && (
                  <span className="text-yellow-400 ml-2">(Test Mode Active)</span>
                )}
              </p>
            </div>
            <div className="flex items-center gap-3">
              <div className="text-right">
                <div className="text-slate-400 text-sm">Est. Translation Time</div>
                <div className="text-2xl font-bold text-white">
                  {user?.credits ? Math.floor(user.credits / 5) : 0} min
                </div>
              </div>
            </div>
          </div>

          {/* Quick Actions */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 pt-8 border-t border-white/10">
            <div className="p-4 rounded-xl bg-white/5 border border-white/10 hover:border-primary-purple/50 transition-colors">
              <div className="text-slate-500 text-xs uppercase tracking-wider mb-1">Available Credits</div>
              <div className="text-2xl font-bold text-white">{user?.credits?.toLocaleString() || 0}</div>
            </div>
            <div className="p-4 rounded-xl bg-white/5 border border-white/10 hover:border-accent-cyan/50 transition-colors">
              <div className="text-slate-500 text-xs uppercase tracking-wider mb-1">Credits Used</div>
              <div className="text-2xl font-bold text-white">0</div>
            </div>
            <div className="p-4 rounded-xl bg-white/5 border border-white/10 hover:border-green-500/50 transition-colors">
              <div className="text-slate-500 text-xs uppercase tracking-wider mb-1">Est. Cost/Min</div>
              <div className="text-2xl font-bold text-white">$0.50</div>
            </div>
          </div>
        </div>
      </div>

      {/* Credit Packages */}
      <div>
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-white font-bold text-lg">Buy More Credits</h3>
          <div className="text-sm text-slate-400">
            {process.env.NEXT_PUBLIC_ENABLE_TEST_PAYMENTS === "true" ? 
              "Test Mode: No real payments required" : 
              "Real Payment Mode: Powered by Polar.sh"}
          </div>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {creditPackages.map((pkg, index) => (
            <motion.div
              key={pkg.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
            >
              <BillingCard package={pkg} />
            </motion.div>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Transaction History */}
        <div className="glass-panel p-6 h-fit">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-white font-bold text-lg">Recent Transactions</h3>
            <button 
              onClick={fetchTransactions}
              disabled={loadingTransactions}
              className="text-sm text-primary-purple-bright hover:text-white transition-colors"
            >
              {loadingTransactions ? "Loading..." : "Refresh"}
            </button>
          </div>
          
          {transactions.length > 0 ? (
            <div className="space-y-1">
              {transactions.slice(0, 5).map((transaction) => (
                <div key={transaction.id} className="flex items-center justify-between p-3 rounded-lg hover:bg-white/5 transition-colors group">
                  <div>
                    <div className="text-white font-medium text-sm">{transaction.description}</div>
                    <div className="text-slate-500 text-xs">
                      {new Date(transaction.created_at).toLocaleDateString()} • 
                      {transaction.credits} credits
                    </div>
                  </div>
                  <div className="flex items-center gap-3">
                    <span className="text-white font-bold">${transaction.amount}</span>
                    <span className={`px-2 py-0.5 rounded-full text-[10px] font-bold uppercase border ${
                      transaction.status === 'completed' 
                        ? 'bg-green-500/10 text-green-400 border-green-500/20'
                        : transaction.status === 'pending'
                        ? 'bg-yellow-500/10 text-yellow-400 border-yellow-500/20'
                        : 'bg-red-500/10 text-red-400 border-red-500/20'
                    }`}>
                      {transaction.status}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-8">
              <CreditCard className="w-12 h-12 text-slate-600 mx-auto mb-3" />
              <p className="text-slate-500">No transactions yet</p>
              <p className="text-slate-600 text-sm mt-1">Purchase credits to see transactions here</p>
            </div>
          )}
        </div>

        {/* Payment Method */}
        <div className="glass-panel p-6">
          <h3 className="text-white font-bold text-lg mb-4">Payment Information</h3>
          
          {/* Test mode notice */}
          {process.env.NEXT_PUBLIC_ENABLE_TEST_PAYMENTS === "true" && (
            <div className="mb-6 p-4 rounded-lg bg-blue-500/10 border border-blue-500/20">
              <div className="flex items-start gap-3">
                <Shield className="w-5 h-5 text-blue-400 flex-shrink-0 mt-0.5" />
                <div>
                  <div className="text-blue-300 font-medium mb-1">Test Mode Active</div>
                  <p className="text-blue-400/80 text-sm">
                    Payments are simulated. No real money will be charged. 
                    Click "Buy" to add test credits instantly.
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Real payment notice */}
          {process.env.NEXT_PUBLIC_ENABLE_TEST_PAYMENTS !== "true" && (
            <div className="mb-6 p-4 rounded-lg bg-purple-500/10 border border-purple-500/20">
              <div className="flex items-start gap-3">
                <CreditCard className="w-5 h-5 text-purple-400 flex-shrink-0 mt-0.5" />
                <div>
                  <div className="text-purple-300 font-medium mb-1">Secure Payment Processing</div>
                  <p className="text-purple-400/80 text-sm">
                    All payments are processed securely through Polar.sh. 
                    After completing payment, your credits will be added automatically.
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Troubleshooting */}
          <div className="p-4 rounded-lg bg-slate-800/50 border border-slate-700">
            <div className="flex items-start gap-3">
              <AlertCircle className="w-5 h-5 text-slate-400 flex-shrink-0 mt-0.5" />
              <div>
                <div className="text-slate-300 font-medium mb-1">Credits not updating?</div>
                <ul className="text-slate-500 text-sm space-y-1">
                  <li>• Click the "Refresh" button next to your balance</li>
                  <li>• Check your transaction history above</li>
                  <li>• Wait a moment for payment processing to complete</li>
                  <li>• Contact support if issues persist</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}