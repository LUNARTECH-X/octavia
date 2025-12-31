"use client";

import { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { useRouter } from 'next/navigation';
import { api } from '@/lib/api';

// Defines the structure of a user, matching what we get from the backend
interface User {
  id: string;
  email: string;
  name: string;
  token: string;
  credits: number;
  verified: boolean;
  created_at?: string;
}

// All the methods and data our user context will provide to components
interface UserContextType {
  user: User | null;
  loading: boolean;
  login: (email: string, password: string) => Promise<{
    success: boolean;
    error?: string;
    requiresVerification?: boolean;
    user?: User;
  }>;
  signup: (email: string, password: string, name: string) => Promise<{
    success: boolean;
    error?: string;
    requiresVerification?: boolean;
    message?: string;
    user_id?: string;
  }>;
  logout: () => Promise<void>;
  updateCredits: (newCredits: number) => void;
  addCredits: (creditsToAdd: number) => Promise<{ success: boolean; newBalance?: number; error?: string }>;
  deductCredits: (creditsToDeduct: number) => Promise<{ success: boolean; newBalance?: number; error?: string }>;
  isAuthenticated: boolean;
  setUser: (user: User | null) => void;
  verifyEmail: (token: string) => Promise<{
    success: boolean;
    error?: string;
    user?: User;
  }>;
  resendVerification: (email: string) => Promise<{
    success: boolean;
    error?: string;
    message?: string
  }>;
  fetchUserProfile: () => Promise<{ success: boolean; error?: string }>;
  refreshCredits: () => Promise<{ success: boolean; credits?: number; error?: string }>;
  demoLogin: () => Promise<{ success: boolean; error?: string; user?: User }>;
}

// Create the context - components will use this to access user data
const UserContext = createContext<UserContextType | undefined>(undefined);

// This provider wraps our entire app and manages all user-related state
export function UserProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const router = useRouter();

  // Check if user is already logged in when the app starts
  useEffect(() => {
    const storedUser = localStorage.getItem('octavia_user');
    if (storedUser) {
      try {
        const parsedUser = JSON.parse(storedUser);
        setUser(parsedUser);

        // Try to refresh the user data to make sure the token is still valid
        fetchUserProfile().catch(() => {
          // If we can't fetch the profile, the token might be expired
          localStorage.removeItem('octavia_user');
          setUser(null);
        });
      } catch (error) {
        console.error('Found corrupted user data in storage:', error);
        localStorage.removeItem('octavia_user');
      }
    }
    setLoading(false);
  }, []);

  // Get the latest user information from the server
  const fetchUserProfile = async () => {
    if (!user) {
      return { success: false, error: 'No user logged in' };
    }

    try {
      const response = await api.getUserProfile();
      if (response.success && response.user) {
        const updatedUser = {
          ...user,
          name: response.user.name,
          credits: response.user.credits,
          verified: response.user.verified,
          created_at: response.user.created_at
        };
        setUser(updatedUser);
        localStorage.setItem('octavia_user', JSON.stringify(updatedUser));
        return { success: true };
      } else {
        return { success: false, error: response.error || 'Failed to fetch profile' };
      }
    } catch (error) {
      console.error('Could not refresh user profile:', error);
      return { success: false, error: 'Network error' };
    }
  };

  // Get the latest credit balance from the server
  const refreshCredits = async () => {
    if (!user) {
      return { success: false, error: 'No user logged in' };
    }

    try {
      // Use getUserProfile as alternative since getUserCredits was removed
      const response = await api.getUserProfile();

      // Safety check for null response
      if (!response) {
        return { success: false, error: 'Failed to fetch user profile' };
      }

      if (response.success && response.user) {
        const updatedUser = {
          ...user,
          credits: response.user.credits,
          // Update other fields too while we're at it
          verified: response.user.verified
        };
        setUser(updatedUser);
        localStorage.setItem('octavia_user', JSON.stringify(updatedUser));
        return { success: true, credits: response.user.credits };
      } else {
        return { success: false, error: response.error || 'Failed to fetch credits' };
      }
    } catch (error) {
      console.error('Could not refresh credits:', error);
      return { success: false, error: 'Network error' };
    }
  };

  // Handle user login with email and password
  const login = async (email: string, password: string) => {
    setLoading(true);
    try {
      const response = await api.login(email, password);

      if (response.success) {
        const userData: User = {
          id: response.user?.id || '',
          email: response.user?.email || email,
          name: response.user?.name || '',
          token: response.token || '',
          credits: response.user?.credits || 0,
          verified: response.user?.verified || false
        };

        setUser(userData);
        localStorage.setItem('octavia_user', JSON.stringify(userData));
        return {
          success: true,
          user: userData
        };
      } else {
        return {
          success: false,
          error: response.error,
          requiresVerification: response.message?.includes('verify') || false
        };
      }
    } catch (error) {
      console.error('Login failed:', error);
      return { success: false, error: 'Network connection failed' };
    } finally {
      setLoading(false);
    }
  };

  // Create a new user account
  const signup = async (email: string, password: string, name: string) => {
    setLoading(true);
    try {
      const response = await api.signup(email, password, name);

      if (response.success) {
        if (response.requires_verification) {
          // User needs to verify their email before they can log in
          localStorage.setItem('pending_verification_email', email);

          return {
            success: true,
            requiresVerification: true,
            message: response.message,
            user_id: response.user?.id
          };
        }

        // Account was created successfully and doesn't need verification
        const userData: User = {
          id: response.user?.id || '',
          email: response.user?.email || email,
          name: response.user?.name || name,
          token: response.token || '',
          credits: response.user?.credits || 0,
          verified: response.user?.verified || true
        };

        setUser(userData);
        localStorage.setItem('octavia_user', JSON.stringify(userData));
        return { success: true };
      } else {
        return { success: false, error: response.error };
      }
    } catch (error) {
      console.error('Signup failed:', error);
      return { success: false, error: 'Network connection failed' };
    } finally {
      setLoading(false);
    }
  };

  // Verify email using the token sent to the user
  const verifyEmail = async (token: string) => {
    setLoading(true);
    try {
      const response = await api.verifyEmail(token);

      if (response.success) {
        const userData: User = {
          id: response.user?.id || '',
          email: response.user?.email || '',
          name: response.user?.name || '',
          token: response.token || '',
          credits: response.user?.credits || 0,
          verified: response.user?.verified || true
        };

        setUser(userData);
        localStorage.setItem('octavia_user', JSON.stringify(userData));
        localStorage.removeItem('pending_verification_email');

        return {
          success: true,
          user: userData
        };
      } else {
        return { success: false, error: response.error };
      }
    } catch (error) {
      console.error('Email verification failed:', error);
      return { success: false, error: 'Network connection failed' };
    } finally {
      setLoading(false);
    }
  };

  // Send a new verification email if the first one was lost or expired
  const resendVerification = async (email: string) => {
    setLoading(true);
    try {
      const response = await api.resendVerification(email);

      if (response.success) {
        return { success: true, message: response.message };
      } else {
        return { success: false, error: response.error };
      }
    } catch (error) {
      console.error('Could not resend verification email:', error);
      return { success: false, error: 'Network connection failed' };
    } finally {
      setLoading(false);
    }
  };

  // Log the user out and clear all session data
  const logout = async () => {
    setLoading(true);
    try {
      await api.logout();
    } catch (error) {
      console.warn('Server logout failed, but clearing local session:', error);
    } finally {
      // Always clear the local data even if server call fails
      setUser(null);
      localStorage.removeItem('octavia_user');
      localStorage.removeItem('pending_verification_email');
      setLoading(false);
      router.push('/login');
    }
  };

  // Update the user's credit balance in the local state
  const updateCredits = (newCredits: number) => {
    if (user && newCredits >= 0) {
      const updatedUser = { ...user, credits: newCredits };
      setUser(updatedUser);
      localStorage.setItem('octavia_user', JSON.stringify(updatedUser));
    }
  };

  // Add credits to the user's account (would normally call backend)
  const addCredits = async (creditsToAdd: number): Promise<{ success: boolean; newBalance?: number; error?: string }> => {
    if (!user) {
      return { success: false, error: 'User not logged in' };
    }

    if (creditsToAdd <= 0) {
      return { success: false, error: 'Credits to add must be positive' };
    }

    try {
      // In a real app, this would call a backend endpoint
      // For now, we just update the local state
      const newBalance = user.credits + creditsToAdd;
      updateCredits(newBalance);

      return {
        success: true,
        newBalance: newBalance
      };
    } catch (error) {
      console.error('Failed to add credits:', error);
      return { success: false, error: 'Failed to add credits' };
    }
  };

  // Deduct credits from the user (for things like video translations)
  const deductCredits = async (creditsToDeduct: number): Promise<{ success: boolean; newBalance?: number; error?: string }> => {
    if (!user) {
      return { success: false, error: 'User not logged in' };
    }

    if (creditsToDeduct <= 0) {
      return { success: false, error: 'Credits to deduct must be positive' };
    }

    if (user.credits < creditsToDeduct) {
      return { success: false, error: 'Insufficient credits' };
    }

    try {
      // In production, this would be handled by the video translation endpoint
      const newBalance = user.credits - creditsToDeduct;
      updateCredits(newBalance);

      return {
        success: true,
        newBalance: newBalance
      };
    } catch (error) {
      console.error('Failed to deduct credits:', error);
      return { success: false, error: 'Failed to deduct credits' };
    }
  };

  // Quick login for testing and demonstration purposes
  const demoLogin = async () => {
    setLoading(true);
    try {
      const response = await api.demoLogin();

      if (response.success) {
        const userData: User = {
          id: response.user?.id || '',
          email: response.user?.email || '',
          name: response.user?.name || '',
          token: response.token || '',
          credits: response.user?.credits || 0,
          verified: response.user?.verified || true
        };

        setUser(userData);
        localStorage.setItem('octavia_user', JSON.stringify(userData));

        return {
          success: true,
          user: userData
        };
      } else {
        return { success: false, error: response.error };
      }
    } catch (error) {
      console.error('Demo login failed:', error);
      return { success: false, error: 'Network connection failed' };
    } finally {
      setLoading(false);
    }
  };

  // This is what gets provided to all components in the app
  const value: UserContextType = {
    user,
    loading,
    login,
    signup,
    logout,
    updateCredits,
    addCredits,
    deductCredits,
    isAuthenticated: !!user,
    setUser,
    verifyEmail,
    resendVerification,
    fetchUserProfile,
    refreshCredits,
    demoLogin
  };

  return (
    <UserContext.Provider value={value}>
      {children}
    </UserContext.Provider>
  );
}

// Custom hook that components use to access the user context
export const useUser = (): UserContextType => {
  const context = useContext(UserContext);
  if (context === undefined) {
    throw new Error('useUser must be used within a UserProvider');
  }
  return context;
};