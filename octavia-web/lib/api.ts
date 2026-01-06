// API service for handling all backend communication
// Manages authentication, video translation, subtitle generation, and user operations
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// Standard response format from all API endpoints
export interface ApiResponse<T = any> {
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
  status?: string;
  progress?: number;
  content?: string;
  segments?: SubtitleSegment[];
  language?: string;
  format?: string;
  segment_count?: number;
  created_at?: string;
  completed_at?: string;
}

export interface User {
  id: string;
  email: string;
  name: string;
  credits: number;
  verified: boolean;
  created_at?: string;
}

// Credit package interface
export interface CreditPackage {
  id: string;
  name: string;
  credits: number;
  price: number;
  description: string;
  features: string[];
  checkout_link?: string;
}

// Payment session response
export interface PaymentSessionResponse {
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
export interface PaymentStatusResponse {
  session_id: string;
  transaction_id: string;
  status: string;
  credits: number;
  description: string;
  created_at: string;
  updated_at: string;
}

// Transaction interface
export interface Transaction {
  id: string;
  amount: number;
  credits: number;
  status: string;
  created_at: string;
  description: string;
  session_id?: string;
  package_id?: string;
}

// Subtitle job interface
export interface SubtitleJobResponse {
  job_id: string;
  download_url?: string;
  format?: string;
  segment_count?: number;
  language?: string;
  success?: boolean;
  message?: string;
}

// Subtitle segment interface
export interface SubtitleSegment {
  id: number;
  start: number;
  end: number;
  text: string;
  confidence?: number;
  words?: Array<{
    word: string;
    start: number;
    end: number;
    confidence: number;
  }>;
}

// Subtitle review data
export interface SubtitleReviewData {
  job_id: string;
  status: string;
  format: string;
  language: string;
  segment_count: number;
  content: string;
  download_url: string;
  created_at: string;
  completed_at?: string;
  segments?: SubtitleSegment[];
}

// Translation request interface
export interface TranslationRequest {
  file: File;
  sourceLanguage: string;
  targetLanguage: string;
}

// Translation response interface
export interface TranslationResponse {
  success: boolean;
  file?: Blob;
  fileName?: string;
  error?: string;
}

// Language option interface
export interface LanguageOption {
  value: string;
  label: string;
  code: string;
}

// Translation progress interface
export interface TranslationProgress {
  status: 'idle' | 'uploading' | 'translating' | 'downloading' | 'complete' | 'error';
  progress: number;
  message?: string;
}

// Available chunk interface for preview
export interface AvailableChunk {
  id: number;
  start_time: number;
  duration: number;
  preview_url: string;
  status: 'completed' | 'processing' | 'failed';
  confidence_score?: number;
  estimated_wer?: number;
  quality_rating?: string;
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

  // Update user token in localStorage
  private updateUserToken(token: string): void {
    if (typeof window === 'undefined') return;

    const userStr = localStorage.getItem('octavia_user');
    if (userStr) {
      try {
        const user = JSON.parse(userStr);
        user.token = token;
        localStorage.setItem('octavia_user', JSON.stringify(user));
      } catch (error) {
        console.error('Failed to update token:', error);
      }
    }
  }

  // Core request handler for all API calls
  private async request<T = any>(
    endpoint: string,
    options: RequestInit = {},
    requiresAuth: boolean = false
  ): Promise<ApiResponse<T>> {
    try {
      const url = `${API_BASE_URL}${endpoint}`;
      console.log(`API Request: ${options.method || 'GET'} ${url}`);

      // Setup headers for the request
      const headers: Record<string, string> = {
        'Accept': 'application/json',
      };

      // Add Authorization header for protected endpoints
      if (requiresAuth) {
        const token = this.getToken();
        console.log(`Auth required, token present: ${!!token}`);
        if (token) {
          headers['Authorization'] = `Bearer ${token}`;
        } else {
          console.log('No token found, returning auth error');
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

      console.log(`Making fetch request to: ${url}`);

      // DEBUG CODE START - ADD THIS
      if (options.body instanceof FormData) {
        console.log('🔍 DEBUG: FormData contents being sent:');
        const formData = options.body as FormData;
        for (const [key, value] of formData.entries()) {
          if (value instanceof File) {
            console.log(`  📁 Field: "${key}" = File: ${value.name} (${value.type}, ${value.size} bytes)`);
          } else {
            console.log(`  📝 Field: "${key}" = "${value}"`);
          }
        }
      }
      // DEBUG CODE END

      let response;
      try {
        // Add timeout to prevent hanging requests (longer for file uploads and translation endpoints)
        const controller = new AbortController();
        const isFileUpload = options.body instanceof FormData;
        const isTranslationEndpoint = endpoint.includes('/translate/') || endpoint.includes('/generate/');
        // 5 minutes for file uploads and translation endpoints, 2 minutes for others
        const timeoutMs = (isFileUpload || isTranslationEndpoint) ? 300000 : 120000;
        const timeoutId = setTimeout(() => controller.abort(new Error(`Request timed out after ${timeoutMs / 1000} seconds`)), timeoutMs);

        response = await fetch(url, {
          ...options,
          headers: {
            ...headers,
            ...options.headers,
          },
          signal: controller.signal,
        });

        clearTimeout(timeoutId);
        console.log(`Fetch completed, response status: ${response.status}`)
      } catch (fetchError) {
        console.error('Fetch failed with network error:', fetchError);
        const isAbortError = fetchError instanceof DOMException && fetchError.name === 'AbortError';
        const errorMessage = isAbortError
          ? `Request timed out. File uploads may take longer - please try again.`
          : (fetchError instanceof Error ? fetchError.message : 'Unknown network error');
        return {
          success: false,
          error: `Network error: ${errorMessage}. Please check your connection and try again.`,
        };
      }

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
              if (errorData.detail) {
                // Handle FastAPI/Pydantic validation errors
                if (Array.isArray(errorData.detail)) {
                  // Extract messages from validation error array
                  errorMessage = errorData.detail.map((err: any) => {
                    if (typeof err === 'string') return err;
                    return err.msg || err.message || JSON.stringify(err);
                  }).join(', ');
                } else if (typeof errorData.detail === 'string') {
                  errorMessage = errorData.detail;
                } else if (typeof errorData.detail === 'object' && errorData.detail.msg) {
                  errorMessage = errorData.detail.msg;
                } else {
                  errorMessage = JSON.stringify(errorData.detail);
                }
              } else if (errorData.error) {
                errorMessage = errorData.error;
              } else if (errorData.message) {
                errorMessage = errorData.message;
              } else if (typeof errorData === 'string') {
                errorMessage = errorData;
              } else {
                errorMessage = `HTTP ${response.status}: ${response.statusText}`;
              }
            } catch {
              // Response wasn't JSON, use text directly
              errorMessage = responseText || errorMessage;
            }
          }
        } catch (parseError) {
          console.error('Failed to parse error response:', parseError);
          errorMessage = `HTTP ${response.status}: ${response.statusText}`;
        }

        // Return the error response
        return {
          success: false,
          error: errorMessage,
          status: response.status.toString(),
        };
      }

      // Parse successful response
      const responseText = await response.text();
      console.log(`Response text received: "${responseText?.substring(0, 200)}..."`);

      let data: any = {};

      if (responseText) {
        try {
          data = JSON.parse(responseText);
          console.log('Parsed response data:', data);
        } catch (parseError) {
          console.error('Server returned invalid JSON:', parseError);
          console.error('Raw response text:', responseText);
          return {
            success: false,
            error: 'Server returned an invalid response format',
          };
        }
      } else {
        console.warn('Server returned empty response');
        return {
          success: false,
          error: 'Server returned an empty response',
        };
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

  // --- SUBTITLE TRANSLATION ENDPOINTS ---

  // Get subtitle translation job status
  async getSubtitleTranslationStatus(jobId: string): Promise<ApiResponse<{
    job_id: string;
    status: string;
    progress: number;
    download_url?: string;
    source_language?: string;
    target_language?: string;
    error?: string;
  }>> {
    return this.request<{
      job_id: string;
      status: string;
      progress: number;
      download_url?: string;
      source_language?: string;
      target_language?: string;
      error?: string;
    }>(`/api/translate/subtitle-status/${jobId}`, {
      method: 'GET',
    }, true);
  }



  async downloadTranslatedSubtitle(jobId: string): Promise<Blob> {
    const token = this.getToken();

    // Try multiple endpoint patterns
    const endpoints = [
      `/api/download/translated-subtitle/${jobId}`,
      `/api/download/subtitle-audio/${jobId}`,
      `/api/generate/subtitle-audio/download/${jobId}`,
      `/api/download/${jobId}`
    ];

    let lastError: Error | null = null;

    for (const endpoint of endpoints) {
      const url = `${API_BASE_URL}${endpoint}`;

      try {
        const response = await fetch(url, {
          headers: token ? {
            'Authorization': `Bearer ${token}`
          } : {},
        });

        if (response.ok) {
          return await response.blob();
        }

        // If not found, try next endpoint
        if (response.status === 404) {
          continue;
        }

        // For other errors, throw immediately
        const errorText = await response.text();
        throw new Error(`Download failed (${response.status}): ${errorText}`);

      } catch (error) {
        lastError = error as Error;
        console.log(`Endpoint ${endpoint} failed, trying next...`);
      }
    }

    // If all endpoints failed
    throw lastError || new Error('Download failed: All endpoints failed');
  }



  async translateSubtitleFile(
    data: TranslationRequest
  ): Promise<ApiResponse<{
    status: string;
    download_url: string;
    source_language: string;
    target_language: string;
    segment_count: number;
    output_path: string;
    remaining_credits: number;
  }>> {
    try {
      // Use query parameters instead of form data for language parameters
      // to avoid FastAPI parameter binding issues with File + Form combination
      const endpoint = `/api/translate/subtitle-file?sourceLanguage=${encodeURIComponent(data.sourceLanguage)}&targetLanguage=${encodeURIComponent(data.targetLanguage)}&format=srt`;

      const formData = new FormData();
      formData.append('file', data.file);

      console.log('DEBUG: Subtitle translation request:', {
        endpoint: endpoint,
        file: data.file.name,
        sourceLanguage: data.sourceLanguage,
        targetLanguage: data.targetLanguage
      });

      return this.request<{
        status: string;
        download_url: string;
        source_language: string;
        target_language: string;
        segment_count: number;
        output_path: string;
        remaining_credits: number;
      }>(endpoint, {
        method: 'POST',
        body: formData,
      }, true);  // Authentication required
    } catch (error) {
      console.error('Translation error:', error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
      };
    }
  }

  // --- AUTHENTICATION ENDPOINTS ---

  // Register a new user account
  async signup(email: string, password: string, name: string): Promise<ApiResponse> {
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
  async login(email: string, password: string): Promise<ApiResponse> {
    const body = JSON.stringify({
      email: email,
      password: password
    });

    const response = await this.request('/api/auth/login', {
      method: 'POST',
      body: body,
    });

    // Store token if login successful
    if (response.success && response.token && response.user) {
      const userData = {
        ...response.user,
        token: response.token
      };

      if (typeof window !== 'undefined') {
        localStorage.setItem('octavia_user', JSON.stringify(userData));
      }
    }

    return response;
  }

  // Logout user and invalidate session
  async logout(): Promise<ApiResponse> {
    const response = await this.request('/api/auth/logout', {
      method: 'POST',
    }, true);

    // Clear localStorage on logout
    if (typeof window !== 'undefined') {
      localStorage.removeItem('octavia_user');
      localStorage.removeItem('last_payment_session');
    }

    return response;
  }

  // --- EMAIL VERIFICATION ---

  // Verify email address using token from verification email
  async verifyEmail(token: string): Promise<ApiResponse> {
    const formData = new FormData();
    formData.append('token', token);

    return this.request('/api/auth/verify', {
      method: 'POST',
      body: formData,
    });
  }

  // Resend verification email to user
  async resendVerification(email: string): Promise<ApiResponse> {
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
  async getCreditPackages(): Promise<ApiResponse<{ packages: CreditPackage[] }>> {
    return this.request<{ packages: CreditPackage[] }>('/api/payments/packages', {
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

        if (response.success && response.data) {
          if (response.data.status === 'completed') {
            return response.data;
          }

          if (response.data.status === 'failed') {
            console.error('Payment failed:', response.data.description);
            return response.data;
          }
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
  async addTestCredits(credits: number): Promise<ApiResponse<{ new_balance: number; credits_added: number }>> {
    const body = JSON.stringify({
      credits: credits
    });

    return this.request<{ new_balance: number; credits_added: number }>('/api/payments/add-test-credits', {
      method: 'POST',
      body: body,
    }, true);
  }

  // Get user profile including credits
  async getUserProfile(): Promise<ApiResponse<User>> {
    return this.request<User>('/api/user/profile', {
      method: 'GET',
    }, true);
  }

  // Get user's transaction history
  async getTransactionHistory(): Promise<ApiResponse<{ transactions: Transaction[] }>> {
    return this.request<{ transactions: Transaction[] }>('/api/payments/transactions', {
      method: 'GET',
    }, true);
  }

  // Get user settings
  async getUserSettings(): Promise<ApiResponse<{
    settings: {
      notifications: {
        translationComplete: boolean;
        emailNotifications: boolean;
        weeklySummary: boolean;
      };
      language: string;
      time_zone: string;
    };
  }>> {
    // For demo account, return mock settings
    const userStr = localStorage.getItem('octavia_user');
    const userData = userStr ? JSON.parse(userStr) : null;
    if (userData?.email === 'demo@octavia.com') {
      return {
        success: true,
        data: {
          settings: {
            notifications: {
              translationComplete: true,
              emailNotifications: false,
              weeklySummary: true,
            },
            language: 'English',
            time_zone: 'UTC (GMT+0)',
          }
        }
      };
    }

    // For real users, try to fetch from API (if endpoint exists)
    try {
      return this.request<{
        settings: {
          notifications: {
            translationComplete: boolean;
            emailNotifications: boolean;
            weeklySummary: boolean;
          };
          language: string;
          time_zone: string;
        };
      }>('/api/user/settings', {
        method: 'GET',
      }, true);
    } catch (error) {
      // Fallback to default settings if endpoint doesn't exist
      return {
        success: true,
        data: {
          settings: {
            notifications: {
              translationComplete: true,
              emailNotifications: false,
              weeklySummary: true,
            },
            language: 'English',
            time_zone: 'UTC (GMT+0)',
          }
        }
      };
    }
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

  // Check status of a translation job
  async getJobStatus(jobId: string): Promise<ApiResponse<{
    job_id: string;
    status: string;
    progress: number;
    type?: string;
    status_message?: string;
    download_url?: string;
    original_filename?: string;
    target_language?: string;
    error?: string;
    chunks_processed?: number;
    total_chunks?: number;
    available_chunks?: AvailableChunk[];
  }>> {
    return this.request<{
      job_id: string;
      status: string;
      progress: number;
      status_message?: string;
      download_url?: string;
      original_filename?: string;
      target_language?: string;
      error?: string;
    }>(`/api/translate/jobs/${jobId}/status`, {
      method: 'GET',
    }, true);
  }

  // Get user's job history
  async getUserJobHistory(): Promise<ApiResponse<{
    jobs: Array<{
      id: string;
      type: string;
      status: string;
      progress: number;
      file_path?: string;
      target_language?: string;
      original_filename?: string;
      created_at: string;
      completed_at?: string;
      result?: any;
      output_path?: string;
      error?: string;
      message?: string;
    }>;
    total: number;
  }>> {
    // For demo account, return mock jobs
    const userStr = localStorage.getItem('octavia_user');
    const userData = userStr ? JSON.parse(userStr) : null;
    if (userData?.email === 'demo@octavia.com') {
      return {
        success: true,
        data: {
          jobs: [
            {
              id: "demo-job-001",
              type: "video",
              status: "completed",
              progress: 100,
              target_language: "es",
              original_filename: "Product Demo Video.mp4",
              created_at: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString(),
              completed_at: new Date(Date.now() - 2 * 60 * 60 * 1000 + 4 * 30 * 1000).toISOString(),
              output_path: "/downloads/demo-job-001.mp4",
              message: "Translation completed successfully"
            },
            {
              id: "demo-job-002",
              type: "audio",
              status: "processing",
              progress: 75,
              target_language: "fr",
              original_filename: "Marketing Podcast Ep. 4.mp3",
              created_at: new Date(Date.now() - 5 * 60 * 60 * 1000).toISOString(),
              message: "Processing audio segments"
            },
            {
              id: "demo-job-003",
              type: "video",
              status: "failed",
              progress: 0,
              target_language: "de",
              original_filename: "Keynote Speech.mp4",
              created_at: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000).toISOString(),
              error: "Unsupported video format"
            }
          ],
          total: 3
        }
      };
    }

    // For real users, fetch from API
    return this.request<{
      jobs: Array<{
        id: string;
        type: string;
        status: string;
        progress: number;
        file_path?: string;
        target_language?: string;
        original_filename?: string;
        created_at: string;
        completed_at?: string;
        result?: any;
        output_path?: string;
        error?: string;
        message?: string;
      }>;
      total: number;
    }>(`/api/translate/jobs/history`, {
      method: 'GET',
    }, true);
  }

  // --- SUBTITLE GENERATION ---

  // Generate subtitles from video/audio file
  async generateSubtitles(
    file: File,
    format: string = 'srt',
    language: string = 'auto'
  ): Promise<ApiResponse<SubtitleJobResponse>> {
    // Use query parameters instead of form data for language/format parameters
    // to avoid FastAPI parameter binding issues with File + Form combination
    const endpoint = `/api/translate/subtitles?language=${encodeURIComponent(language)}&format=${encodeURIComponent(format)}`;

    const formData = new FormData();
    formData.append('file', file);

    console.log('DEBUG: Subtitle generation request:', {
      endpoint: endpoint,
      file: file.name,
      language: language,
      format: format
    });

    return this.request<SubtitleJobResponse>(endpoint, {
      method: 'POST',
      body: formData,
    }, true);  // Authentication required
  }

  // Translate audio file
  async translateAudio(params: {
    file: File;
    sourceLanguage: string;
    targetLanguage: string;
  }): Promise<ApiResponse<{ job_id: string }>> {
    const formData = new FormData();
    formData.append('file', params.file);
    formData.append('source_lang', params.sourceLanguage);
    formData.append('target_lang', params.targetLanguage);

    return this.request<{ job_id: string }>(`/api/translate/audio`, {
      method: 'POST',
      body: formData,
    }, true);
  }

  // Get subtitle job status (uses generic job status endpoint)
  async getSubtitleJobStatus(jobId: string): Promise<ApiResponse<{
    job_id: string;
    status: string;
    progress: number;
    download_url?: string;
    format?: string;
    segment_count?: number;
    language?: string;
    error?: string;
  }>> {
    return this.getJobStatus(jobId);
  }

  // Get subtitle data for review
  async getSubtitleReviewData(jobId: string): Promise<ApiResponse<SubtitleReviewData>> {
    return this.request<SubtitleReviewData>(`/api/translate/subtitles/review/${jobId}`, {
      method: 'GET',
    }, true);
  }

  // Download subtitle file
  async downloadSubtitleFile(jobId: string): Promise<Blob> {
    const token = this.getToken();
    const url = `${API_BASE_URL}/api/translate/download/subtitles/${jobId}`;

    const response = await fetch(url, {
      headers: token ? {
        'Authorization': `Bearer ${token}`
      } : {},
    });

    if (!response.ok) {
      throw new Error(`Download failed: ${response.status} ${response.statusText}`);
    }

    return await response.blob();
  }

  // --- POLLING HELPERS ---

  // Download a specific translated chunk for preview
  async downloadChunk(jobId: string, chunkId: number): Promise<Blob> {
    const token = this.getToken();

    const url = `${API_BASE_URL}/api/download/chunk/${jobId}/${chunkId}`;

    const response = await fetch(url, {
      headers: token ? {
        'Authorization': `Bearer ${token}`
      } : {},
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Chunk download failed: ${response.statusText}`);
    }

    return await response.blob();
  }

  // Poll job status until completed or failed
  async pollJobStatus(
    jobId: string,
    interval: number = 2000,
    maxAttempts: number = 300
  ): Promise<ApiResponse<{
    job_id: string;
    status: string;
    progress: number;
    status_message?: string;
    download_url?: string;
    original_filename?: string;
    target_language?: string;
    error?: string;
    chunks_processed?: number;
    total_chunks?: number;
    available_chunks?: AvailableChunk[];
  }>> {
    for (let attempt = 0; attempt < maxAttempts; attempt++) {
      try {
        const response = await this.getJobStatus(jobId);

        if (response.success && response.data) {
          if (response.data.status === 'completed' || response.data.status === 'failed') {
            return response;
          }
        }

        // Wait before next attempt
        await new Promise(resolve => setTimeout(resolve, interval));
      } catch (error) {
        console.error('Error polling job status:', error);
      }
    }

    return {
      success: false,
      error: 'Timeout waiting for job completion',
    };
  }

  // Poll subtitle generation status
  async pollSubtitleStatus(
    jobId: string,
    interval: number = 2000,
    maxAttempts: number = 60
  ): Promise<ApiResponse> {
    for (let attempt = 0; attempt < maxAttempts; attempt++) {
      try {
        const response = await this.getSubtitleJobStatus(jobId);

        if (response.success && response.data) {
          if (response.data.status === 'completed' || response.data.status === 'failed') {
            return response;
          }
        }

        // Wait before next attempt
        await new Promise(resolve => setTimeout(resolve, interval));
      } catch (error) {
        console.error('Error polling subtitle status:', error);
      }
    }

    return {
      success: false,
      error: 'Timeout waiting for subtitle generation',
    };
  }

  // --- FILE DOWNLOAD ---

  // Download translated video file by job ID
  async downloadFile(jobId: string): Promise<Blob> {
    const token = this.getToken();

    // Try endpoints in order of priority
    const endpoints = [
      `/api/translate/download/video/${jobId}`,  // Translation routes endpoint (has job in memory)
      `/api/download/video/${jobId}`,             // App.py video endpoint
      `/api/download/${jobId}`,                   // Generic fallback
    ];

    for (const endpoint of endpoints) {
      const url = `${API_BASE_URL}${endpoint}`;
      console.log(`Trying download endpoint: ${url}`);

      try {
        const response = await fetch(url, {
          headers: token ? {
            'Authorization': `Bearer ${token}`
          } : {},
        });

        if (response.ok) {
          console.log(`Download successful from: ${endpoint}`);
          return await response.blob();
        }

        console.log(`Endpoint ${endpoint} returned ${response.status}`);
        
        // Continue to next endpoint on 404
        if (response.status === 404) {
          continue;
        }
      } catch (error) {
        console.log(`Endpoint ${endpoint} failed:`, error);
        continue;
      }
    }

    // All endpoints failed - try to get job status for debugging
    try {
      const jobStatusResponse = await this.getJobStatus(jobId);
      console.log('Job status for debugging:', jobStatusResponse);
    } catch (statusError) {
      console.error('Failed to get job status for debugging:', statusError);
    }

    throw new Error(`Download failed: All endpoints returned 404 for job ${jobId}`);
  }

  // Download any file by URL
  async downloadFileByUrl(url: string): Promise<Blob> {
    const token = this.getToken();

    const response = await fetch(url, {
      headers: token ? {
        'Authorization': `Bearer ${token}`
      } : {},
    });

    if (!response.ok) {
      throw new Error(`Download failed: ${response.statusText}`);
    }

    return await response.blob();
  }

  // Save downloaded file with proper filename
  saveFile(blob: Blob, filename: string): void {
    if (typeof window === 'undefined') return;

    const downloadUrl = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = downloadUrl;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(downloadUrl);
  }

  // --- SYSTEM HEALTH ---

  // Check if backend API is reachable
  async healthCheck(): Promise<ApiResponse> {
    return this.request('/api/health');
  }

  // Login with demo/test account
  async demoLogin(): Promise<ApiResponse> {
    return this.request('/api/auth/demo-login', {
      method: 'POST',
    });
  }

  // Test basic API connectivity
  async testConnection(): Promise<ApiResponse> {
    return this.request('/');
  }

  // Direct signup test (bypasses request wrapper for debugging)
  async testSignupDirect(email: string, password: string, name: string): Promise<any> {
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

  // Language options for translation
  getLanguageOptions(): LanguageOption[] {
    return [
      { value: 'english', label: 'English', code: 'en' },
      { value: 'spanish', label: 'Spanish', code: 'es' },
      { value: 'french', label: 'French', code: 'fr' },
      { value: 'german', label: 'German', code: 'de' },
      { value: 'italian', label: 'Italian', code: 'it' },
      { value: 'portuguese', label: 'Portuguese', code: 'pt' },
      { value: 'russian', label: 'Russian', code: 'ru' },
      { value: 'japanese', label: 'Japanese', code: 'ja' },
      { value: 'korean', label: 'Korean', code: 'ko' },
      { value: 'chinese', label: 'Chinese', code: 'zh-cn' },
      { value: 'arabic', label: 'Arabic', code: 'ar' },
      { value: 'hindi', label: 'Hindi', code: 'hi' },
    ];
  }

  // Store payment session for later polling
  storePaymentSession(sessionId: string, transactionId: string, packageId: string): void {
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
  getStoredPaymentSession(): {
    session_id: string;
    transaction_id: string;
    package_id: string;
    timestamp: number
  } | null {
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
  clearStoredPaymentSession(): void {
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

  // Store subtitle job data
  storeSubtitleJob(jobId: string, data: any): void {
    if (typeof window === 'undefined') return;

    const jobData = {
      job_id: jobId,
      ...data,
      timestamp: Date.now()
    };

    localStorage.setItem(`subtitle_job_${jobId}`, JSON.stringify(jobData));
  }

  // Get stored subtitle job data
  getStoredSubtitleJob(jobId: string): any {
    if (typeof window === 'undefined') return null;

    const jobData = localStorage.getItem(`subtitle_job_${jobId}`);
    if (!jobData) return null;

    try {
      return JSON.parse(jobData);
    } catch (error) {
      console.error('Failed to parse stored subtitle job:', error);
      localStorage.removeItem(`subtitle_job_${jobId}`);
      return null;
    }
  }

  // Clear stored subtitle job data
  clearStoredSubtitleJob(jobId: string): void {
    if (typeof window === 'undefined') return;
    localStorage.removeItem(`subtitle_job_${jobId}`);
  }

  // Format SRT timestamp
  formatSRTTimestamp(seconds: number): string {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    const milliseconds = Math.floor((seconds - Math.floor(seconds)) * 1000);

    return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')},${milliseconds.toString().padStart(3, '0')}`;
  }

  // Parse SRT content
  parseSRTContent(srtText: string): SubtitleSegment[] {
    const segments: SubtitleSegment[] = [];
    const lines = srtText.split('\n');

    let currentSegment: Partial<SubtitleSegment> = {};
    let textBuffer: string[] = [];

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i].trim();

      if (!line) {
        // Empty line indicates end of segment
        if (currentSegment.id && textBuffer.length > 0) {
          segments.push({
            id: currentSegment.id!,
            start: currentSegment.start!,
            end: currentSegment.end!,
            text: textBuffer.join(' ').trim()
          });
        }
        currentSegment = {};
        textBuffer = [];
        continue;
      }

      if (!currentSegment.id && /^\d+$/.test(line)) {
        // Segment number
        currentSegment.id = parseInt(line, 10);
      } else if (currentSegment.id && !currentSegment.start && line.includes('-->')) {
        // Timestamp line
        const [startStr, endStr] = line.split('-->').map(s => s.trim());
        currentSegment.start = this.parseSRTTimestamp(startStr);
        currentSegment.end = this.parseSRTTimestamp(endStr);
      } else if (currentSegment.id && currentSegment.start) {
        // Text line
        textBuffer.push(line);
      }
    }

    // Add last segment if exists
    if (currentSegment.id && textBuffer.length > 0) {
      segments.push({
        id: currentSegment.id!,
        start: currentSegment.start!,
        end: currentSegment.end!,
        text: textBuffer.join(' ').trim()
      });
    }

    return segments;
  }

  // Parse SRT timestamp to seconds
  private parseSRTTimestamp(timestamp: string): number {
    const [time, milliseconds] = timestamp.split(',');
    const [hours, minutes, seconds] = time.split(':').map(Number);
    const ms = milliseconds ? parseInt(milliseconds, 10) / 1000 : 0;

    return hours * 3600 + minutes * 60 + seconds + ms;
  }

  // Generate dummy subtitle data for testing
  generateDummySubtitles(count: number = 10): SubtitleSegment[] {
    const segments: SubtitleSegment[] = [];
    let currentTime = 0;

    for (let i = 1; i <= count; i++) {
      const duration = 3 + Math.random() * 4; // 3-7 seconds per segment
      const segment: SubtitleSegment = {
        id: i,
        start: currentTime,
        end: currentTime + duration,
        text: `This is dummy subtitle text for segment ${i}. It simulates generated speech recognition output.`,
        confidence: 0.8 + Math.random() * 0.2 // 80-100% confidence
      };

      segments.push(segment);
      currentTime += duration + 0.5; // Add small gap between segments
    }

    return segments;
  }
}

export const api = new ApiService();

// Helper function to safely access API response properties
export const safeApiResponse = <T>(response: ApiResponse<T> | null, defaultValue: T): T => {
  if (!response || !response.data) {
    return defaultValue;
  }
  return response.data as T;
};

// Helper function to check if response is successful
export const isSuccess = (response: ApiResponse | null): response is ApiResponse => {
  return response !== null && response.success === true;
};
