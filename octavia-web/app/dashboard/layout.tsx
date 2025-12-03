// app/dashboard/layout.tsx
import { Sidebar } from "@/components/dashboard/Sidebar";
import { ProtectedRoute } from "@/components/ProtectedRoute";

export default function DashboardLayout({
    children,
}: {
    children: React.ReactNode;
}) {
    return (
        <ProtectedRoute>
            <div className="flex h-screen min-h-screen w-full flex-row overflow-hidden bg-bg-dark text-white font-sans selection:bg-primary-purple/30">
                <Sidebar />

                <main className="flex-1 overflow-y-auto p-8 custom-scrollbar relative">
                    {/* Ambient Background Glows */}
                    <div
                        className="glow-purple-strong"
                        style={{ width: "600px", height: "600px", position: "fixed", top: "-200px", right: "-100px", zIndex: 0 }}
                    />
                    <div
                        className="glow-purple"
                        style={{ width: "400px", height: "400px", position: "fixed", bottom: "-100px", left: "100px", zIndex: 0 }}
                    />

                    <div className="relative z-10 mx-auto max-w-6xl">
                        {children}
                    </div>
                </main>
            </div>
        </ProtectedRoute>
    );
}