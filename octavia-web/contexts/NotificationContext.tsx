"use client";

import React, { createContext, useContext, useState, useCallback, ReactNode } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { CheckCircle, XCircle, AlertCircle, Info, X } from "lucide-react";

type NotificationType = "success" | "error" | "warning" | "info";

interface Notification {
    id: string;
    type: NotificationType;
    title?: string;
    message: string;
    duration?: number;
}

interface NotificationContextType {
    showNotification: (params: Omit<Notification, "id">) => void;
    hideNotification: (id: string) => void;
}

const NotificationContext = createContext<NotificationContextType | undefined>(undefined);

export const useNotification = () => {
    const context = useContext(NotificationContext);
    if (!context) {
        throw new Error("useNotification must be used within a NotificationProvider");
    }
    return context;
};

export const NotificationProvider = ({ children }: { children: ReactNode }) => {
    const [notifications, setNotifications] = useState<Notification[]>([]);

    const hideNotification = useCallback((id: string) => {
        setNotifications((prev) => prev.filter((n) => n.id !== id));
    }, []);

    const showNotification = useCallback(({ type, title, message, duration = 5000 }: Omit<Notification, "id">) => {
        const id = Math.random().toString(36).substring(2, 9);
        const newNotification = { id, type, title, message, duration };

        setNotifications((prev) => [...prev, newNotification]);

        if (duration > 0) {
            setTimeout(() => {
                hideNotification(id);
            }, duration);
        }
    }, [hideNotification]);

    return (
        <NotificationContext.Provider value={{ showNotification, hideNotification }}>
            {children}
            <div className="fixed bottom-6 right-6 z-[9999] flex flex-col gap-4 w-full max-w-sm pointer-events-none">
                <AnimatePresence>
                    {notifications.map((notification) => (
                        <NotificationItem
                            key={notification.id}
                            notification={notification}
                            onClose={() => hideNotification(notification.id)}
                        />
                    ))}
                </AnimatePresence>
            </div>
        </NotificationContext.Provider>
    );
};

const NotificationItem = ({ notification, onClose }: { notification: Notification; onClose: () => void }) => {
    const icons = {
        success: <CheckCircle className="w-5 h-5 text-cyan-400" />,
        error: <XCircle className="w-5 h-5 text-red-400" />,
        warning: <AlertCircle className="w-5 h-5 text-amber-400" />,
        info: <Info className="w-5 h-5 text-blue-400" />,
    };

    const borders = {
        success: "border-cyan-500/20",
        error: "border-red-500/20",
        warning: "border-amber-500/20",
        info: "border-blue-500/20",
    };

    const glows = {
        success: "shadow-[0_0_20px_rgba(6,182,212,0.1)]",
        error: "shadow-[0_0_20px_rgba(239,68,68,0.1)]",
        warning: "shadow-[0_0_20px_rgba(245,158,11,0.1)]",
        info: "shadow-[0_0_20px_rgba(59,130,246,0.1)]",
    };

    return (
        <motion.div
            initial={{ opacity: 0, x: 50, scale: 0.9 }}
            animate={{ opacity: 1, x: 0, scale: 1 }}
            exit={{ opacity: 0, x: 20, scale: 0.95 }}
            className={`pointer-events-auto glass-panel p-4 flex gap-4 items-start ${borders[notification.type]} ${glows[notification.type]}`}
        >
            <div className="flex-shrink-0 mt-0.5">
                {icons[notification.type]}
            </div>
            <div className="flex-1 min-w-0">
                {notification.title && (
                    <h4 className="text-white font-bold text-sm mb-1">{notification.title}</h4>
                )}
                <p className="text-slate-300 text-sm leading-relaxed">{notification.message}</p>
            </div>
            <button
                onClick={onClose}
                className="flex-shrink-0 text-slate-500 hover:text-white transition-colors"
            >
                <X className="w-4 h-4" />
            </button>
        </motion.div>
    );
};
