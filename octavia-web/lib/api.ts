// API service for handling all backend communication
// Manages authentication, video translation, and user operations
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// Standard response format from all API endpoints
interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
  user?: any;
  token?: string;
  job_id?: string;
  download_url?: string;
  remaining_credits?: number;
  requires_verification?: boolean;
}

interface User {
  id: string;
  email: string;
  name: string;
  credits: number;
  verified: boolean;
  created_at?: string;
}

// Credit package interface
interface CreditPackage {
  id: string;
  name: string;
  credits: number;
  price: number;
  description: string;
  features: string[];
  checkout_link?: string;
}

// Payment session response
interface PaymentSessionResponse {
  session_id: string;
  transaction_id: string;
  checkout_url: string;
  package_id: string;
  credits: number;
  price: number;
  message: string;
  status: string;
  test_mode?: boolean;
  new_balance?: number;
  credits_added?: number;
}

// Payment status response
interface PaymentStatusResponse {
  session_id: string;
  transaction_id: string;
  status: string;
  credits: number;
  description: string;
  created_at: string;
  updated_at: string;
}

// Transaction interface
interface Transaction {
  id: string;
  amount: number;
  credits: number;
  status: string;
  created_at: string;
  description: string;
  session_id?: string;
  package_id?: string;
}

class ApiService {
  // Get authentication token from localStorage
  private getToken(): string | null {
    if (typeof window === 'undefined') return null;
    
    const userStr = localStorage.getItem('octavia_user');
    if (userStr) {
      try {
        const user = JSON.parse(userStr);
        return user.token || null;
      } catch (error) {
        console.error('Failed to parse user token from localStorage:', error);
        return null;
      }
    }
    return null;
  }

  // Core request handler for all API calls
  private async request<T = any>(
    endpoint: string,
    options: RequestInit = {},
    requiresAuth: boolean = false
  ): Promise<ApiResponse<T>> {
    try {
      const url = `${API_BASE_URL}${endpoint}`;
      
      // Setup headers for the request
      const headers: Record<string, string> = {
        'Accept': 'application/json',
      };
      
      // Add Authorization header for protected endpoints
      if (requiresAuth) {
        const token = this.getToken();
        if (token) {
          headers['Authorization'] = `Bearer ${token}`;
        } else {
          return {
            success: false,
            error: 'Authentication required. Please log in.',
          };
        }
      }
      
      // Set Content-Type for JSON payloads (skip for FormData)
      const isFormData = options.body instanceof FormData;
      if (!isFormData && options.body) {
        headers['Content-Type'] = 'application/json';
      }
      
      const response = await fetch(url, {
        ...options,
        headers: {
          ...headers,
          ...options.headers,
        },
      });

      // Handle HTTP errors (4xx, 5xx responses)
      if (!response.ok) {
        let errorData: any = {};
        let errorMessage = `HTTP ${response.status}: ${response.statusText}`;
        
        try {
          // Try to extract error message from response body
          const responseText = await response.text();
          
          if (responseText) {
            try {
              errorData = JSON.parse(responseText);
              
              // Look for error message in common response fields
              errorMessage = errorData.detail || 
                           errorData.error || 
                           errorData.message || 
                           (typeof errorData === 'string' ? errorData : errorMessage);
            } catch {
              // Response wasn't JSON, use text directly
              errorMessage = responseText || errorMessage;
            }
          }
        } catch (parseError) {
          console.error('Failed to parse error response:', parseError);
          errorMessage = `HTTP ${response.status}: ${response.statusText}`;
        }
        
        throw new Error(errorMessage);
      }

      // Parse successful response
      const responseText = await response.text();
      let data: any = {};
      
      if (responseText) {
        try {
          data = JSON.parse(responseText);
        } catch (parseError) {
          console.error('Server returned invalid JSON:', parseError);
          throw new Error('Server returned an invalid response format');
        }
      }
      
      return data;
    } catch (error) {
      console.error('API request failed:', error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'An unexpected error occurred',
      };
    }
  }

  // --- AUTHENTICATION ENDPOINTS ---

  // Register a new user account
  async signup(email: string, password: string, name: string) {
    const body = JSON.stringify({
      email: email,
      password: password,
      name: name
    });

    return this.request('/api/auth/signup', {
      method: 'POST',
      body: body,
    });
  }

  // Login existing user and receive authentication token
  async login(email: string, password: string) {
    const body = JSON.stringify({
      email: email,
      password: password
    });

    return this.request('/api/auth/login', {
      method: 'POST',
      body: body,
    });
  }

  // Logout user and invalidate session
  async logout() {
    const token = this.getToken();
    
    const response = await this.request('/api/auth/logout', {
      method: 'POST',
      headers: token ? {
        'Authorization': `Bearer ${token}`
      } : {},
    }, false);
    
    // Clear localStorage on logout
    if (typeof window !== 'undefined') {
      localStorage.removeItem('octavia_user');
      localStorage.removeItem('last_payment_session');
    }
    
    return response;
  }

  // --- EMAIL VERIFICATION ---

  // Verify email address using token from verification email
  async verifyEmail(token: string) {
    const formData = new FormData();
    formData.append('token', token);

    return this.request('/api/auth/verify', {
      method: 'POST',
      body: formData,
    });
  }

  // Resend verification email to user
  async resendVerification(email: string) {
    const body = JSON.stringify({
      email: email
    });

    return this.request('/api/auth/resend-verification', {
      method: 'POST',
      body: body,
    });
  }

  // --- PAYMENT & CREDITS ---

  // Get available credit packages
  async getCreditPackages() {
    return this.request<{
      packages: CreditPackage[];
    }>('/api/payments/packages', {
      method: 'GET',
    });
  }

  // Create payment session for credit purchase
  async createPaymentSession(packageId: string): Promise<ApiResponse<PaymentSessionResponse>> {
    const body = JSON.stringify({
      package_id: packageId
    });

    return this.request<PaymentSessionResponse>('/api/payments/create-session', {
      method: 'POST',
      body: body,
    }, true);
  }

  // Check payment status
  async checkPaymentStatus(sessionId: string): Promise<ApiResponse<PaymentStatusResponse>> {
    return this.request<PaymentStatusResponse>(`/api/payments/status/${sessionId}`, {
      method: 'GET',
    }, true);
  }

  // Poll payment status until completed or failed
  async pollPaymentStatus(sessionId: string, maxAttempts: number = 30): Promise<PaymentStatusResponse | null> {
    for (let attempt = 0; attempt < maxAttempts; attempt++) {
      try {
        const response = await this.checkPaymentStatus(sessionId);
        
        if (response.success && response.status === 'completed') {
          return response;
        }
        
        if (response.success && response.status === 'failed') {
          console.error('Payment failed:', response.description);
          return response;
        }
        
        // Wait before next attempt
        await new Promise(resolve => setTimeout(resolve, 1000));
      } catch (error) {
        console.error('Error polling payment status:', error);
      }
    }
    
    return null;
  }

  // Add test credits (for development/testing without payment)
  async addTestCredits(credits: number) {
    const body = JSON.stringify({
      credits: credits
    });

    return this.request<{
      new_balance: number;
      credits_added: number;
    }>('/api/payments/add-test-credits', {
      method: 'POST',
      body: body,
    }, true);
  }

  // Get user's transaction history
  async getTransactionHistory(): Promise<ApiResponse<{ transactions: Transaction[] }>> {
    return this.request<{ transactions: Transaction[] }>('/api/payments/transactions', {
      method: 'GET',
    }, true);
  }

  // Debug webhook endpoint
  async debugWebhook(): Promise<ApiResponse<{
    transactions: any[];
    webhook_secret_configured: boolean;
    test_mode: boolean;
    polar_server: string;
  }>> {
    return this.request('/api/payments/webhook/debug', {
      method: 'GET',
    }, true);
  }

  // --- VIDEO TRANSLATION ---

  // Upload video file for translation to target language
  async translateVideo(
    file: File,
    targetLanguage: string = 'es',
    userEmail: string
  ) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('target_language', targetLanguage);

    return this.request('/api/translate/video', {
      method: 'POST',
      body: formData,
    }, true);
  }

  // Check status of a translation job
  async getJobStatus(jobId: string) {
    return this.request<{
      job_id: string;
      status: string;
      progress: number;
      status_message?: string;
      download_url?: string;
      original_filename?: string;
      target_language?: string;
      error?: string;
    }>(`/api/jobs/${jobId}/status`, {
      method: 'GET',
    }, true);
  }

  // --- USER PROFILE ---

  // Get current user's profile information
  async getUserProfile(): Promise<ApiResponse<{ user: User }>> {
    return this.request<{ user: User }>('/api/user/profile', {
      method: 'GET',
    }, true);
  }

  // Get user's current credit balance
  async getUserCredits(): Promise<ApiResponse<{ credits: number; email: string }>> {
    return this.request<{ credits: number; email: string }>('/api/user/credits', {
      method: 'GET',
    }, true);
  }

  // --- FILE DOWNLOAD ---

  // Download translated video file by job ID
  async downloadFile(jobId: string): Promise<Blob> {
    const token = this.getToken();
    const url = `${API_BASE_URL}/api/download/${jobId}`;
    
    const response = await fetch(url, {
      headers: token ? {
        'Authorization': `Bearer ${token}`
      } : {},
    });
    
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Download failed: ${response.statusText}`);
    }
    
    return await response.blob();
  }

  // --- SYSTEM HEALTH ---

  // Check if backend API is reachable
  async healthCheck() {
    return this.request('/api/health');
  }

  // Login with demo/test account
  async demoLogin() {
    return this.request('/api/auth/demo-login', {
      method: 'POST',
    });
  }

  // Test basic API connectivity
  async testConnection() {
    return this.request('/');
  }

  // Direct signup test (bypasses request wrapper for debugging)
  async testSignupDirect(email: string, password: string, name: string) {
    try {
      const response = await fetch(`${API_BASE_URL}/api/auth/signup`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
        },
        body: JSON.stringify({ email, password, name }),
      });
      
      const responseText = await response.text();
      
      if (!response.ok) {
        throw new Error(`Signup failed: ${response.status} ${response.statusText}`);
      }
      
      let data = {};
      if (responseText) {
        try {
          data = JSON.parse(responseText);
        } catch (e) {
          console.error('Response was not valid JSON:', e);
          data = { raw: responseText };
        }
      }
      
      return data;
    } catch (error) {
      console.error('Direct signup test error:', error);
      throw error;
    }
  }

  // --- HELPER METHODS ---

  // Store payment session for later polling
  storePaymentSession(sessionId: string, transactionId: string, packageId: string) {
    if (typeof window === 'undefined') return;
    
    const paymentData = {
      session_id: sessionId,
      transaction_id: transactionId,
      package_id: packageId,
      timestamp: Date.now()
    };
    
    localStorage.setItem('last_payment_session', JSON.stringify(paymentData));
  }

  // Get stored payment session
  getStoredPaymentSession() {
    if (typeof window === 'undefined') return null;
    
    const paymentData = localStorage.getItem('last_payment_session');
    if (!paymentData) return null;
    
    try {
      const parsed = JSON.parse(paymentData);
      
      // Check if session is older than 5 minutes
      const timeElapsed = Date.now() - parsed.timestamp;
      if (timeElapsed > 5 * 60 * 1000) {
        localStorage.removeItem('last_payment_session');
        return null;
      }
      
      return parsed;
    } catch (error) {
      console.error('Failed to parse stored payment session:', error);
      localStorage.removeItem('last_payment_session');
      return null;
    }
  }

  // Clear stored payment session
  clearStoredPaymentSession() {
    if (typeof window === 'undefined') return;
    localStorage.removeItem('last_payment_session');
  }

  // Check URL parameters for payment success
  checkUrlForPaymentSuccess(): { success: boolean; sessionId: string | null } {
    if (typeof window === 'undefined') return { success: false, sessionId: null };
    
    const urlParams = new URLSearchParams(window.location.search);
    const paymentSuccess = urlParams.get('payment_success');
    const sessionId = urlParams.get('session_id');
    
    if (paymentSuccess === 'true' && sessionId) {
      // Clean URL
      const newUrl = window.location.pathname;
      window.history.replaceState({}, document.title, newUrl);
      
      return { success: true, sessionId };
    }
    
    return { success: false, sessionId: null };
  }
}

export const api = new ApiService();