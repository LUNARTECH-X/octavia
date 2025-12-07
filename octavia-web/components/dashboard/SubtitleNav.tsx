"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Upload, Clock, Eye } from "lucide-react";

export function SubtitleNav() {
  const pathname = usePathname();
  
  const navItems = [
    {
      name: "Generate",
      href: "/dashboard/subtitles",
      icon: Upload
    },
    {
      name: "Progress",
      href: "/dashboard/subtitles/progress",
      icon: Clock
    },
    {
      name: "Review",
      href: "/dashboard/subtitles/review",
      icon: Eye
    }
  ];

  return (
    <div className="border-b border-white/10 bg-[#0D0221]/80 backdrop-blur-lg">
      <div className="px-6 py-4">
        <div className="flex items-center gap-8">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = pathname === item.href;
            
            return (
              <Link
                key={item.href}
                href={item.href}
                className={`flex flex-col items-center gap-1.5 p-2 rounded-lg transition-all ${
                  isActive
                    ? "text-primary-purple-bright bg-primary-purple/10 border border-primary-purple/20"
                    : "text-slate-400 hover:text-white hover:bg-white/5"
                }`}
              >
                <Icon className="w-5 h-5" />
                <span className="text-sm font-medium">{item.name}</span>
              </Link>
            );
          })}
        </div>
      </div>
    </div>
  );
}
