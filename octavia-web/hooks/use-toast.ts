"use client";

import * as React from "react";

export type ToastProps = {
  title?: string;
  description?: string;
  variant?: "default" | "destructive";
  duration?: number;
};

export function useToast() {
  const [toasts, setToasts] = React.useState<Array<ToastProps & { id: string }>>([]);

  const toast = React.useCallback((props: ToastProps) => {
    const id = Math.random().toString(36).substring(2, 9);
    const newToast = { ...props, id };
    
    setToasts((prevToasts) => [...prevToasts, newToast]);

    // Auto remove after duration
    const duration = props.duration || 5000;
    setTimeout(() => {
      setToasts((prevToasts) => prevToasts.filter((toast) => toast.id !== id));
    }, duration);

    return { id };
  }, []);

  return {
    toast,
    toasts,
    dismiss: (id: string) => {
      setToasts((prevToasts) => prevToasts.filter((toast) => toast.id !== id));
    },
  };
}