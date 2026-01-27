"use client";

import { useState, useEffect } from "react";
import { User, Lock, Shield, Mail, Camera, Save, Loader2 } from "lucide-react";
import { useUser } from "@/contexts/UserContext";
import { api } from "@/lib/api";

export default function ProfilePage() {
  const { user, loading: userLoading, fetchUserProfile } = useUser();
  const [isEditing, setIsEditing] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [formData, setFormData] = useState({
    name: "",
    email: "",
    company: "",
    role: ""
  });

  // Initialize form data with user data
  useEffect(() => {
    if (user) {
      setFormData({
        name: user.name || "",
        email: user.email || "",
        company: "LunarTech", // Default value
        role: "User" // Default value based on credits or other logic
      });
    }
  }, [user]);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleSaveChanges = async () => {
    if (!user) return;

    setIsSaving(true);
    try {
      const response = await api.updateUserProfile({
        name: formData.name,
        email: formData.email
      });

      if (response.success) {
        // Refresh user profile from server
        await fetchUserProfile();
        setIsEditing(false);
        alert("Profile updated successfully!");
      } else {
        throw new Error(response.error || "Failed to update profile");
      }
    } catch (error) {
      console.error("Failed to update profile:", error);
      alert("Failed to update profile. Please try again.");
    } finally {
      setIsSaving(false);
    }
  };

  const handleChangePassword = () => {
    const currentPassword = prompt("Enter your current password:");
    if (!currentPassword) return;

    const newPassword = prompt("Enter your new password (minimum 6 characters):");
    if (!newPassword) return;

    if (newPassword.length < 6) {
      alert("Password must be at least 6 characters long.");
      return;
    }

    const confirmPassword = prompt("Confirm your new password:");
    if (confirmPassword !== newPassword) {
      alert("Passwords do not match.");
      return;
    }

    // Call API to change password
    api.changePassword(currentPassword, newPassword)
      .then(response => {
        if (response.success) {
          alert("Password changed successfully!");
        } else {
          throw new Error(response.error || "Failed to change password");
        }
      })
      .catch(error => {
        console.error("Failed to change password:", error);
        alert("Failed to change password. Please check your current password and try again.");
      });
  };

  const handleDeleteAccount = async () => {
    const confirmText = prompt("Type 'DELETE' to confirm account deletion:");
    if (confirmText !== "DELETE") {
      alert("Account deletion cancelled.");
      return;
    }

    if (confirm("Are you absolutely sure? This will permanently delete your account and all associated data. This action cannot be undone.")) {
      try {
        const response = await api.deleteUserAccount();
        if (response.success) {
          alert("Account deletion initiated. Your account will be permanently deleted in 30 days. You will now be logged out.");
          // Force logout
          window.location.href = "/login";
        } else {
          throw new Error(response.error || "Failed to delete account");
        }
      } catch (error) {
        console.error("Failed to delete account:", error);
        alert("Failed to delete account. Please try again or contact support.");
      }
    }
  };

  const getUserRole = () => {
    if (!user) return "User";

    // Determine role based on credits or other criteria
    if (user.credits >= 5000) return "Premium";
    if (user.credits >= 1000) return "Pro";
    if (user.credits >= 100) return "Member";
    return "User";
  };

  const getLastPasswordChange = () => {
    // This would come from the backend in a real app
    return "3 months ago";
  };

  if (userLoading) {
    return (
      <div className="space-y-8">
        <div>
          <h1 className="font-display text-3xl font-black text-white mb-2 text-glow-purple">Profile & Security</h1>
          <p className="text-slate-400 text-sm">Loading user data...</p>
        </div>
        <div className="glass-panel p-8 flex items-center justify-center">
          <Loader2 className="w-8 h-8 text-primary-purple-bright animate-spin" />
        </div>
      </div>
    );
  }

  if (!user) {
    return (
      <div className="space-y-8">
        <div>
          <h1 className="font-display text-3xl font-black text-white mb-2 text-glow-purple">Profile & Security</h1>
          <p className="text-slate-400 text-sm">Please log in to view your profile</p>
        </div>
        <div className="glass-panel p-8 text-center">
          <User className="w-12 h-12 text-slate-400 mx-auto mb-4" />
          <h3 className="text-white text-lg font-bold mb-2">No User Found</h3>
          <p className="text-slate-400">Please log in to access your profile information.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex flex-wrap justify-between items-center gap-4">
        <div className="flex items-center gap-5">
          <div className="flex flex-col">
            <h2 className="text-white text-2xl font-black leading-tight bg-gradient-to-r from-white via-primary-purple-bright to-white bg-clip-text text-transparent drop-shadow-[0_0_10px_rgba(168,85,247,0.4)]">
              Octavia
            </h2>
            <p className="text-[10px] font-bold leading-tight tracking-[0.2em] bg-gradient-to-r from-primary-purple-bright via-accent-cyan to-primary-purple-bright bg-clip-text text-transparent text-glow-purple">
              RISE BEYOND LANGUAGE
            </p>
          </div>

          <div className="h-10 w-[1px] bg-white/10 hidden sm:block mx-1" />

          <div>
            <h1 className="font-display text-3xl font-black text-white text-glow-purple">Profile & Security</h1>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Left Column: Personal Info */}
        <div className="lg:col-span-2 space-y-6">
          {/* Profile Card */}
          <div className="glass-panel p-8">
            <div className="flex items-center gap-6 mb-8">
              <div className="relative group cursor-pointer">
                <div className="w-24 h-24 rounded-full bg-white/10 border-2 border-white/20 flex items-center justify-center overflow-hidden">
                  {user.name ? (
                    <div className="w-full h-full flex items-center justify-center text-2xl font-bold text-primary-purple-bright bg-white/5">
                      {user.name.charAt(0).toUpperCase()}
                    </div>
                  ) : (
                    <User className="w-10 h-10 text-slate-400" />
                  )}
                </div>
                <div className="absolute inset-0 bg-black/50 rounded-full flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity">
                  <Camera className="w-6 h-6 text-white" />
                </div>
              </div>
              <div>
                <h2 className="text-2xl font-bold text-white">{user.name || "User"}</h2>
                <p className="text-slate-400">{user.email}</p>
                <div className="flex items-center gap-3 mt-2">
                  <div className="inline-flex items-center gap-2 px-2 py-1 rounded-md bg-primary-purple/10 border border-primary-purple/20 text-primary-purple-bright text-xs font-medium">
                    <Shield className="w-3 h-3" />
                    {getUserRole()}
                  </div>
                  <div className="text-xs text-slate-500">
                    Credits: <span className="font-bold text-primary-purple-bright">{user.credits}</span>
                  </div>
                  <div className="text-xs text-slate-500">
                    Status: <span className={`font-bold ${user.verified ? 'text-accent-cyan' : 'text-yellow-400'}`}>
                      {user.verified ? 'Verified' : 'Unverified'}
                    </span>
                  </div>
                </div>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-2">
                <label className="text-sm font-medium text-slate-300">Full Name</label>
                <input
                  name="name"
                  type="text"
                  value={formData.name}
                  onChange={handleInputChange}
                  disabled={!isEditing}
                  className="glass-input w-full disabled:opacity-50 disabled:cursor-not-allowed"
                />
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium text-slate-300">Email Address</label>
                <div className="relative">
                  <Mail className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
                  <input
                    name="email"
                    type="email"
                    value={formData.email}
                    onChange={handleInputChange}
                    disabled={!isEditing}
                    className="glass-input w-full pl-10 disabled:opacity-50 disabled:cursor-not-allowed"
                  />
                </div>
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium text-slate-300">Company</label>
                <input
                  name="company"
                  type="text"
                  value={formData.company}
                  onChange={handleInputChange}
                  disabled={!isEditing}
                  className="glass-input w-full disabled:opacity-50 disabled:cursor-not-allowed"
                />
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium text-slate-300">Role</label>
                <input
                  name="role"
                  type="text"
                  value={formData.role}
                  onChange={handleInputChange}
                  disabled={!isEditing}
                  className="glass-input w-full disabled:opacity-50 disabled:cursor-not-allowed"
                />
              </div>
            </div>

            <div className="mt-8 flex justify-end gap-3">
              {isEditing ? (
                <>
                  <button
                    onClick={() => setIsEditing(false)}
                    disabled={isSaving}
                    className="px-6 py-2 rounded-lg border border-white/10 hover:bg-white/5 text-sm text-slate-300 hover:text-white transition-colors disabled:opacity-50"
                  >
                    Cancel
                  </button>
                  <button
                    onClick={handleSaveChanges}
                    disabled={isSaving}
                    className="btn-border-beam disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    <div className="btn-border-beam-inner px-6 py-2 flex items-center gap-2">
                      {isSaving ? (
                        <>
                          <Loader2 className="w-4 h-4 animate-spin" />
                          Saving...
                        </>
                      ) : (
                        <>
                          <Save className="w-4 h-4" />
                          Save Changes
                        </>
                      )}
                    </div>
                  </button>
                </>
              ) : (
                <button
                  onClick={() => setIsEditing(true)}
                  className="btn-border-beam"
                >
                  <div className="btn-border-beam-inner px-6 py-2">
                    Edit Profile
                  </div>
                </button>
              )}
            </div>
          </div>
        </div>

        {/* Right Column: Security */}
        <div className="space-y-6">
          <div className="glass-panel p-6">
            <div className="flex items-center gap-3 mb-6">
              <div className="p-2 rounded-lg bg-primary-purple/10 text-primary-purple-bright">
                <Lock className="w-5 h-5" />
              </div>
              <h3 className="text-white font-bold text-lg">Security</h3>
            </div>

            <div className="space-y-6">
              <div>
                <h4 className="text-sm font-medium text-white mb-2">Password</h4>
                <p className="text-xs text-slate-400 mb-3">Last changed {getLastPasswordChange()}</p>
                <button
                  onClick={handleChangePassword}
                  className="w-full py-2 rounded-lg border border-white/10 hover:bg-white/5 text-sm text-slate-300 hover:text-white transition-colors"
                >
                  Change Password
                </button>
              </div>

              <div className="pt-6 border-t border-white/5">
                <div className="flex items-center justify-between mb-2">
                  <h4 className="text-sm font-medium text-white">Two-Factor Auth</h4>
                  <span className="text-xs text-accent-cyan font-medium">Enabled</span>
                </div>
                <p className="text-xs text-slate-400 mb-3">Secure your account with 2FA.</p>
                <button className="w-full py-2 rounded-lg border border-white/10 hover:bg-white/5 text-sm text-slate-300 hover:text-white transition-colors">
                  Configure
                </button>
              </div>
            </div>
          </div>

          <div className="glass-panel p-6 border-red-500/20">
            <h3 className="text-red-400 font-bold text-lg mb-2">Danger Zone</h3>
            <p className="text-slate-400 text-xs mb-4">
              Permanently delete your account and all of your content.
            </p>
            <button
              onClick={handleDeleteAccount}
              className="w-full py-2 rounded-lg bg-red-500/10 hover:bg-red-500/20 text-red-400 text-sm font-medium transition-colors border border-red-500/20"
            >
              Delete Account
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
