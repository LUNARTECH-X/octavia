"use client";

import { motion, AnimatePresence } from "framer-motion";
import { AlertTriangle, X, Loader2 } from "lucide-react";

interface ConfirmationModalProps {
    isOpen: boolean;
    onClose: () => void;
    onConfirm: () => void;
    title?: string;
    message?: string;
    confirmText?: string;
    cancelText?: string;
    isLoading?: boolean;
    variant?: "danger" | "warning" | "info";
}

export default function ConfirmationModal({
    isOpen,
    onClose,
    onConfirm,
    title = "Confirm Action",
    message = "Are you sure you want to proceed?",
    confirmText = "Confirm",
    cancelText = "Cancel",
    isLoading = false,
    variant = "danger",
}: ConfirmationModalProps) {
    const variantStyles = {
        danger: {
            icon: "bg-red-500/20 border-red-500/30",
            iconColor: "text-red-400",
            button: "bg-red-500 hover:bg-red-600 text-white",
            glow: "shadow-[0_0_30px_rgba(239,68,68,0.3)]",
        },
        warning: {
            icon: "bg-yellow-500/20 border-yellow-500/30",
            iconColor: "text-yellow-400",
            button: "bg-yellow-500 hover:bg-yellow-600 text-black",
            glow: "shadow-[0_0_30px_rgba(234,179,8,0.3)]",
        },
        info: {
            icon: "bg-blue-500/20 border-blue-500/30",
            iconColor: "text-blue-400",
            button: "bg-blue-500 hover:bg-blue-600 text-white",
            glow: "shadow-[0_0_30px_rgba(59,130,246,0.3)]",
        },
    };

    const styles = variantStyles[variant];

    return (
        <AnimatePresence>
            {isOpen && (
                <>
                    {/* Backdrop */}
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        onClick={!isLoading ? onClose : undefined}
                        className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50"
                    />

                    {/* Modal */}
                    <motion.div
                        initial={{ opacity: 0, scale: 0.9, y: 20 }}
                        animate={{ opacity: 1, scale: 1, y: 0 }}
                        exit={{ opacity: 0, scale: 0.9, y: 20 }}
                        transition={{ type: "spring", damping: 25, stiffness: 300 }}
                        className="fixed left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 z-50 w-full max-w-md"
                    >
                        <div className={`glass-panel p-6 relative overflow-hidden ${styles.glow}`}>
                            {/* Glass shine effect */}
                            <div className="absolute inset-0 bg-gradient-to-br from-white/5 to-transparent pointer-events-none" />

                            {/* Close button */}
                            <button
                                onClick={onClose}
                                disabled={isLoading}
                                className="absolute top-4 right-4 p-1 text-slate-400 hover:text-white transition-colors disabled:opacity-50"
                            >
                                <X className="w-5 h-5" />
                            </button>

                            {/* Content */}
                            <div className="relative z-10 flex flex-col items-center text-center">
                                {/* Icon */}
                                <div className={`w-16 h-16 rounded-2xl ${styles.icon} border flex items-center justify-center mb-4`}>
                                    <AlertTriangle className={`w-8 h-8 ${styles.iconColor}`} />
                                </div>

                                {/* Title */}
                                <h3 className="text-xl font-bold text-white mb-2">{title}</h3>

                                {/* Message */}
                                <p className="text-slate-400 text-sm mb-6 max-w-sm">{message}</p>

                                {/* Buttons */}
                                <div className="flex items-center gap-3 w-full">
                                    <button
                                        onClick={onClose}
                                        disabled={isLoading}
                                        className="flex-1 px-4 py-3 rounded-lg bg-white/5 border border-white/10 text-slate-300 hover:bg-white/10 hover:text-white transition-all disabled:opacity-50"
                                    >
                                        {cancelText}
                                    </button>
                                    <button
                                        onClick={onConfirm}
                                        disabled={isLoading}
                                        className={`flex-1 px-4 py-3 rounded-lg font-medium transition-all disabled:opacity-50 flex items-center justify-center gap-2 ${styles.button}`}
                                    >
                                        {isLoading ? (
                                            <>
                                                <Loader2 className="w-4 h-4 animate-spin" />
                                                <span>Cancelling...</span>
                                            </>
                                        ) : (
                                            confirmText
                                        )}
                                    </button>
                                </div>
                            </div>
                        </div>
                    </motion.div>
                </>
            )}
        </AnimatePresence>
    );
}
