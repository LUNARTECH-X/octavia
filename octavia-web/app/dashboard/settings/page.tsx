//in the settings page i added a logout button that calls the logout function from UserContext and shows a loading state while logging out
// app/dashboard/settings/page.tsx
"use client";

import { Bell, Globe, Shield, Palette, Save, LogOut, User, Key, Trash2, Loader2 } from "lucide-react";
import { useUser } from "@/contexts/UserContext";
import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { api } from "@/lib/api";

export default function SettingsPage() {
    const { user, logout } = useUser();
    const [isLoggingOut, setIsLoggingOut] = useState(false);
    const router = useRouter();

    const handleLogout = async () => {
        setIsLoggingOut(true);
        try {
            await logout();
            // logout function already redirects to /login
        } catch (error) {
            console.error("Logout failed:", error);
        } finally {
            setIsLoggingOut(false);
        }
    };

    const [notifications, setNotifications] = useState({
        translationComplete: true,
        emailNotifications: false,
        weeklySummary: true,
    });

    const [language, setLanguage] = useState("English");
    const [timeZone, setTimeZone] = useState("UTC (GMT+0)");
    const [isSaving, setIsSaving] = useState(false);
    const [loading, setLoading] = useState(true);

    const loadSettings = async () => {
        try {
            console.log("Loading settings for user:", user?.email);

            // Check if user is authenticated
            if (!user) {
                console.log("No user found, using default settings");
                setNotifications({
                    translationComplete: true,
                    emailNotifications: false,
                    weeklySummary: true,
                });
                setLanguage("English");
                setTimeZone("UTC (GMT+0)");
                return;
            }

            const response = await api.getUserSettings();
            console.log("Settings response:", response);

            // Handle null/undefined response
            if (!response) {
                console.error("Settings API returned null/undefined response");
                // Use default settings
                setNotifications({
                    translationComplete: true,
                    emailNotifications: false,
                    weeklySummary: true,
                });
                setLanguage("English");
                setTimeZone("UTC (GMT+0)");
                return;
            }

            if (response.success && response.data?.settings) {
                const settings = response.data.settings;
                setNotifications({
                    translationComplete: settings.notifications?.translationComplete ?? true,
                    emailNotifications: settings.notifications?.emailNotifications ?? false,
                    weeklySummary: settings.notifications?.weeklySummary ?? true,
                });
                setLanguage(settings.language || "English");
                setTimeZone(settings.time_zone || "UTC (GMT+0)");
            } else if (response.error) {
                console.error("Failed to load settings:", response.error);
                // Use default settings on error
                setNotifications({
                    translationComplete: true,
                    emailNotifications: false,
                    weeklySummary: true,
                });
                setLanguage("English");
                setTimeZone("UTC (GMT+0)");
            } else {
                console.error("Unexpected response structure:", response);
                // Use default settings for unexpected responses
                setNotifications({
                    translationComplete: true,
                    emailNotifications: false,
                    weeklySummary: true,
                });
                setLanguage("English");
                setTimeZone("UTC (GMT+0)");
            }
        } catch (error) {
            console.error("Failed to load settings:", error);
            // Use default settings on exception
            setNotifications({
                translationComplete: true,
                emailNotifications: false,
                weeklySummary: true,
            });
            setLanguage("English");
            setTimeZone("UTC (GMT+0)");
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        loadSettings();
    }, []);

    const handleSaveChanges = async () => {
        setIsSaving(true);
        try {
            const settingsData = {
                notifications,
                language,
                time_zone: timeZone
            };

            const response = await api.updateUserSettings(settingsData);
            if (response.success) {
                alert("Settings saved successfully!");
            } else {
                throw new Error(response.message || "Failed to save settings");
            }
        } catch (error) {
            console.error("Failed to save settings:", error);
            alert("Failed to save settings. Please try again.");
        } finally {
            setIsSaving(false);
        }
    };

    const handleChangePassword = () => {
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

        // In a real implementation, you would also ask for current password
        const currentPassword = prompt("Enter your current password:");
        if (!currentPassword) return;

        // Call API to change password
        api.changePassword(currentPassword, newPassword)
            .then(response => {
                if (response.success) {
                    alert("Password changed successfully!");
                } else {
                    throw new Error(response.message || "Failed to change password");
                }
            })
            .catch(error => {
                console.error("Failed to change password:", error);
                alert("Failed to change password. Please check your current password and try again.");
            });
    };

    if (loading) {
        return (
            <div className="space-y-8">
                <div>
                    <h1 className="font-display text-3xl font-black text-white text-glow-purple mb-2">Settings</h1>
                    <p className="text-slate-400 text-sm">Loading settings...</p>
                </div>
                <div className="glass-panel p-8 flex items-center justify-center">
                    <Loader2 className="w-8 h-8 text-primary-purple-bright animate-spin" />
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
                        <h1 className="font-display text-3xl font-black text-white text-glow-purple">Settings</h1>
                    </div>
                </div>
            </div>

            <div className="max-w-4xl mx-auto w-full space-y-6">
                {/* User Info Section */}
                <div className="glass-panel p-6">
                    <h2 className="text-xl font-bold text-white mb-4">Account Information</h2>
                    <div className="space-y-4">
                        {user && (
                            <div className="space-y-3">
                                <div className="flex items-center justify-between">
                                    <div>
                                        <p className="text-white font-medium">Name</p>
                                        <p className="text-slate-400 text-sm">{user.name}</p>
                                    </div>
                                    <button className="text-primary-purple-bright text-sm font-medium hover:text-primary-purple transition-colors">
                                        Edit
                                    </button>
                                </div>
                                <div className="flex items-center justify-between">
                                    <div>
                                        <p className="text-white font-medium">Email</p>
                                        <p className="text-slate-400 text-sm">{user.email}</p>
                                    </div>
                                    <button className="text-primary-purple-bright text-sm font-medium hover:text-primary-purple transition-colors">
                                        Change
                                    </button>
                                </div>
                                <div className="flex items-center justify-between">
                                    <div>
                                        <p className="text-white font-medium">Credits</p>
                                        <p className="text-slate-400 text-sm">{user.credits} remaining</p>
                                    </div>
                                    <button
                                        onClick={() => router.push('/dashboard/billing')}
                                        className="text-primary-purple-bright text-sm font-medium hover:text-primary-purple transition-colors"
                                    >
                                        Buy More
                                    </button>
                                </div>
                            </div>
                        )}
                    </div>
                </div>

                {/* Notifications Section */}
                <div className="glass-panel p-6">
                    <h2 className="text-xl font-bold text-white mb-4">Notification Preferences</h2>
                    <div className="space-y-4">
                        <label className="flex items-center justify-between cursor-pointer group">
                            <div>
                                <p className="text-white font-medium">Translation Complete</p>
                                <p className="text-slate-400 text-xs">Receive notifications when translations finish</p>
                            </div>
                            <input
                                type="checkbox"
                                className="form-checkbox rounded border-white/20 bg-white/5 text-primary-purple focus:ring-primary-purple focus:ring-offset-0 size-5"
                                checked={notifications.translationComplete}
                                onChange={(e) => setNotifications({ ...notifications, translationComplete: e.target.checked })}
                            />
                        </label>

                        <label className="flex items-center justify-between cursor-pointer group">
                            <div>
                                <p className="text-white font-medium">Email Notifications</p>
                                <p className="text-slate-400 text-xs">Send email updates for completed jobs</p>
                            </div>
                            <input
                                type="checkbox"
                                className="form-checkbox rounded border-white/20 bg-white/5 text-primary-purple focus:ring-primary-purple focus:ring-offset-0 size-5"
                                checked={notifications.emailNotifications}
                                onChange={(e) => setNotifications({ ...notifications, emailNotifications: e.target.checked })}
                            />
                        </label>

                        <label className="flex items-center justify-between cursor-pointer group">
                            <div>
                                <p className="text-white font-medium">Weekly Summary</p>
                                <p className="text-slate-400 text-xs">Receive a weekly report of your activity</p>
                            </div>
                            <input
                                type="checkbox"
                                className="form-checkbox rounded border-white/20 bg-white/5 text-primary-purple focus:ring-primary-purple focus:ring-offset-0 size-5"
                                checked={notifications.weeklySummary}
                                onChange={(e) => setNotifications({ ...notifications, weeklySummary: e.target.checked })}
                            />
                        </label>
                    </div>
                </div>

                {/* Language & Region */}
                <div className="glass-panel p-6">
                    <h2 className="text-xl font-bold text-white mb-4">Language & Region</h2>
                    <div className="space-y-4">
                        <div>
                            <label className="text-white text-sm font-semibold mb-2 block">Interface Language</label>
                            <select
                                className="glass-select w-full"
                                value={language}
                                onChange={(e) => setLanguage(e.target.value)}
                            >
                                <option>English</option>
                                <option>Spanish</option>
                                <option>French</option>
                                <option>German</option>
                            </select>
                        </div>

                        <div>
                            <label className="text-white text-sm font-semibold mb-2 block">Time Zone</label>
                            <select
                                className="glass-select w-full"
                                value={timeZone}
                                onChange={(e) => setTimeZone(e.target.value)}
                            >
                                <option>UTC (GMT+0)</option>
                                <option>EST (GMT-5)</option>
                                <option>PST (GMT-8)</option>
                                <option>CET (GMT+1)</option>
                            </select>
                        </div>
                    </div>
                </div>

                {/* Account Actions */}
                <div className="glass-panel p-6 border border-red-500/20">
                    <h2 className="text-xl font-bold text-white mb-4">Account Actions</h2>
                    <div className="space-y-4">
                        <div className="flex items-center justify-between">
                            <div>
                                <p className="text-white font-medium">Change Password</p>
                                <p className="text-slate-400 text-xs">Update your account password</p>
                            </div>
                            <button className="flex items-center gap-2 px-4 py-2 bg-white/5 border border-white/10 text-slate-300 rounded-lg hover:bg-white/10 transition-all">
                                <Key className="w-4 h-4" />
                                <span className="text-sm">Change</span>
                            </button>
                        </div>

                        <div className="flex items-center justify-between">
                            <div>
                                <p className="text-white font-medium">Delete Account</p>
                                <p className="text-slate-400 text-xs">Permanently delete your account and all data</p>
                            </div>
                            <button className="flex items-center gap-2 px-4 py-2 bg-red-500/10 border border-red-500/20 text-red-400 rounded-lg hover:bg-red-500/20 transition-all">
                                <Trash2 className="w-4 h-4" />
                                <span className="text-sm">Delete</span>
                            </button>
                        </div>

                        <div className="flex items-center justify-between pt-4 border-t border-white/10">
                            <div>
                                <p className="text-white font-medium">Logout</p>
                                <p className="text-slate-400 text-xs">Sign out of your account</p>
                            </div>
                            <button
                                onClick={handleLogout}
                                disabled={isLoggingOut}
                                className="flex items-center gap-2 px-4 py-2 bg-red-500/10 border border-red-500/20 text-red-400 rounded-lg hover:bg-red-500/20 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                            >
                                <LogOut className={`w-4 h-4 ${isLoggingOut ? 'animate-pulse' : ''}`} />
                                <span className="text-sm">
                                    {isLoggingOut ? "Logging out..." : "Logout"}
                                </span>
                            </button>
                        </div>
                    </div>
                </div>

                {/* Save Button */}
                <button
                    onClick={handleSaveChanges}
                    className="btn-border-beam w-full group"
                >
                    <div className="btn-border-beam-inner flex items-center justify-center gap-2 py-3">
                        <Save className="w-5 h-5" />
                        <span>Save Changes</span>
                    </div>
                </button>
            </div>
        </div>
    );
}
