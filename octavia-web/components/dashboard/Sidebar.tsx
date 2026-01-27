// components/dashboard/Sidebar.tsx - Updated with logout button
"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { cn } from "@/lib/utils";
import {
    LayoutGrid,
    Mic,
    CreditCard,
    History,
    User,
    Users,
    LogOut,
    HelpCircle,
    LifeBuoy,
    Settings,
    Folder
} from "lucide-react";
import { useUser } from "@/contexts/UserContext";
import { api } from "@/lib/api";
import { useState } from "react";

const navItems = [
    { name: "New Translation", href: "/dashboard", icon: LayoutGrid, primary: true },
    { name: "My Voices", href: "/dashboard/voices", icon: Mic },
    { name: "Projects", href: "/dashboard/projects", icon: Folder },
    { name: "Job History", href: "/dashboard/history", icon: History },
    { name: "Billing", href: "/dashboard/billing", icon: CreditCard },
    { name: "Team", href: "/dashboard/team", icon: Users },
    { name: "Settings", href: "/dashboard/settings", icon: Settings },
];

const bottomItems = [
    { name: "Profile", href: "/dashboard/profile", icon: User },
    { name: "Help", href: "/dashboard/help", icon: HelpCircle },
    { name: "Support", href: "/dashboard/support", icon: LifeBuoy },
];

export function Sidebar() {
    const pathname = usePathname();
    const { user, logout, setUser } = useUser();
    const router = useRouter();
    const [isLoggingOut, setIsLoggingOut] = useState(false);

    const handleLogout = async () => {
        setIsLoggingOut(true);
        try {
            // Option 1: Use the context's logout function
            await logout();

            // Option 2: Or call the API directly
            // await api.logout();

            // Option 3: Or manually clear state
            // setUser(null);
            // localStorage.removeItem('user');
            // localStorage.removeItem('auth_token');

            // Redirect to login
            router.push("/login");

        } catch (error) {
            console.error("Logout failed:", error);
            // Fallback: Clear local state anyway
            setUser(null);
            localStorage.removeItem('user');
            localStorage.removeItem('auth_token');
            router.push("/login");
        } finally {
            setIsLoggingOut(false);
        }
    };

    return (
        <div className="flex h-full w-64 flex-col justify-between border-r border-white/10 bg-[#0D0221]/50 backdrop-blur-xl p-4">
            <div className="flex flex-col gap-6">
                {/* Logo Area */}
                <div className="flex items-center gap-3 px-2 py-2 group cursor-default">
                    <div className="relative w-8 h-8 flex items-center justify-center shrink-0">
                        <img
                            src="/lunartech_logo_small.png"
                            alt="LunarTech Logo"
                            className="w-full h-full object-contain relative z-10 transition-all duration-500 group-hover:scale-110 group-hover:rotate-12"
                        />
                        {/* Multi-layered purple glows */}
                        <div className="absolute inset-0 bg-primary-purple-bright/60 blur-xl rounded-full animate-pulse" />
                        <div className="absolute inset-0 bg-primary-purple/40 blur-lg rounded-full opacity-75 group-hover:opacity-100 transition-opacity duration-300" />
                        <div className="absolute inset-0 bg-accent-cyan/30 blur-md rounded-full opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
                    </div>
                    <div className="flex flex-col">
                        <h1 className="text-white text-lg font-black tracking-[0.15em] bg-gradient-to-r from-white via-primary-purple-bright to-white bg-clip-text text-transparent drop-shadow-[0_0_8px_rgba(168,85,247,0.3)]">
                            LUNARTECH
                        </h1>
                        <p className="text-[10px] font-bold leading-tight tracking-widest text-slate-500">
                            MASTER BRAND
                        </p>
                    </div>
                </div>

                {/* User Info */}
                {user && (
                    <div className="glass-card p-3">
                        <div className="text-sm text-white font-medium">{user.name}</div>
                        <div className="text-xs text-slate-400">{user.email}</div>
                        <div className="flex items-center gap-2 mt-2">
                            <div className="w-2 h-2 rounded-full bg-accent-cyan animate-pulse" />
                            <div className="text-xs text-slate-300">
                                Credits: <span className="font-bold text-primary-purple-bright">{user.credits}</span>
                            </div>
                        </div>
                    </div>
                )}

                {/* Main Nav */}
                <div className="flex flex-col gap-2">
                    {navItems.map((item) => {
                        const isActive = pathname === item.href;
                        const Icon = item.icon;

                        return (
                            <Link
                                key={item.href}
                                href={item.href}
                                className={cn(
                                    "flex items-center gap-3 px-3 py-2.5 rounded-lg transition-all duration-200 group",
                                    item.primary
                                        ? "border border-primary-purple/30 bg-primary-purple/10 hover:bg-primary-purple/20"
                                        : "hover:bg-white/5",
                                    isActive ? "bg-white/10 text-white" : "text-slate-400 hover:text-white"
                                )}
                            >
                                <Icon
                                    className={cn(
                                        "w-5 h-5 transition-colors",
                                        item.primary ? "text-primary-purple-bright" : "group-hover:text-white",
                                        isActive ? "text-primary-purple-bright" : "text-slate-500"
                                    )}
                                />
                                <span className={cn(
                                    "text-sm font-medium leading-normal",
                                    item.primary && "text-primary-purple-bright"
                                )}>
                                    {item.name}
                                </span>
                            </Link>
                        );
                    })}
                </div>
            </div>

            {/* Bottom Nav */}
            <div className="flex flex-col gap-1.5 pt-4 border-t border-white/5">
                {bottomItems.map((item) => {
                    const isActive = pathname === item.href;
                    const Icon = item.icon;

                    return (
                        <Link
                            key={item.href}
                            href={item.href}
                            className={cn(
                                "flex items-center gap-3 px-3 py-2 rounded-lg transition-all duration-200 hover:bg-white/5 group",
                                isActive ? "bg-white/10 text-white" : "text-slate-500 hover:text-slate-300"
                            )}
                        >
                            <Icon className={cn(
                                "w-4 h-4 transition-colors",
                                isActive ? "text-white" : "group-hover:text-slate-300"
                            )} />
                            <span className="text-xs font-medium leading-normal">
                                {item.name}
                            </span>
                        </Link>
                    );
                })}

                {/* Logout Button */}
                <button
                    onClick={handleLogout}
                    disabled={isLoggingOut}
                    className="flex w-full items-center gap-3 px-3 py-2 text-red-400 hover:bg-red-500/10 hover:text-red-300 rounded-lg transition-all duration-200 group mt-1 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                    <LogOut className={cn(
                        "w-4 h-4 transition-colors",
                        isLoggingOut ? "animate-pulse" : "group-hover:text-red-300"
                    )} />
                    <span className="text-xs font-medium leading-normal">
                        {isLoggingOut ? "Logging out..." : "Logout"}
                    </span>
                </button>

                {/* Version Info */}
                <div className="pt-2 mt-2 border-t border-white/5">
                    <div className="flex items-center justify-between px-1">
                        <span className="text-[10px] text-slate-600 font-medium">v3.0.0</span>
                        <span className="text-[10px] text-slate-600">LunarTech</span>
                    </div>
                </div>
            </div>
        </div>
    );
}