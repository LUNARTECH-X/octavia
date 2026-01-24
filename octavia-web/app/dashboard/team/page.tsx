"use client";

import { motion } from "framer-motion";
import { Users, UserPlus, MoreVertical, Shield, Mail } from "lucide-react";

const teamMembers = [
    { id: 1, name: "Alex Morgan", email: "alex.morgan@example.com", role: "Admin", status: "Active", avatar: "AM" },
    { id: 2, name: "Sarah Chen", email: "sarah.chen@example.com", role: "Editor", status: "Active", avatar: "SC" },
    { id: 3, name: "Mike Ross", email: "mike.ross@example.com", role: "Viewer", status: "Pending", avatar: "MR" },
];

export default function TeamPage() {
    return (
        <div className="space-y-8">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="font-display text-3xl font-black text-white mb-2 text-glow-purple">Team & Organization</h1>
                    <p className="text-slate-400 text-sm">Manage your team members and permissions</p>
                </div>
                <button className="btn-border-beam">
                    <div className="btn-border-beam-inner flex items-center gap-2 px-6 py-2.5">
                        <UserPlus className="w-4 h-4" />
                        <span>Invite Member</span>
                    </div>
                </button>
            </div>

            {/* Team Stats */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
                <div className="glass-panel p-5 flex items-center gap-4">
                    <div className="w-12 h-12 rounded-xl bg-primary-purple/10 flex items-center justify-center text-primary-purple-bright">
                        <Users className="w-6 h-6" />
                    </div>
                    <div>
                        <div className="text-2xl font-bold text-white">12</div>
                        <div className="text-xs text-slate-400 uppercase tracking-wider">Total Members</div>
                    </div>
                </div>
                <div className="glass-panel p-5 flex items-center gap-4">
                    <div className="w-12 h-12 rounded-xl bg-accent-cyan/10 flex items-center justify-center text-accent-cyan">
                        <Shield className="w-6 h-6" />
                    </div>
                    <div>
                        <div className="text-2xl font-bold text-white">3</div>
                        <div className="text-xs text-slate-400 uppercase tracking-wider">Admins</div>
                    </div>
                </div>
                <div className="glass-panel p-5 flex items-center gap-4">
                    <div className="w-12 h-12 rounded-xl bg-blue-500/10 flex items-center justify-center text-blue-400">
                        <Mail className="w-6 h-6" />
                    </div>
                    <div>
                        <div className="text-2xl font-bold text-white">2</div>
                        <div className="text-xs text-slate-400 uppercase tracking-wider">Pending Invites</div>
                    </div>
                </div>
            </div>

            {/* Members List */}
            <div className="glass-panel overflow-hidden">
                <div className="p-6 border-b border-white/5">
                    <h3 className="text-lg font-bold text-white">Members</h3>
                </div>
                <div className="divide-y divide-white/5">
                    {teamMembers.map((member) => (
                        <div key={member.id} className="p-4 flex items-center justify-between hover:bg-white/5 transition-colors group">
                            <div className="flex items-center gap-4">
                                <div className="w-10 h-10 rounded-full bg-white/10 flex items-center justify-center text-sm font-bold text-white border border-white/10">
                                    {member.avatar}
                                </div>
                                <div>
                                    <div className="font-medium text-white">{member.name}</div>
                                    <div className="text-xs text-slate-500">{member.email}</div>
                                </div>
                            </div>

                            <div className="flex items-center gap-6">
                                <span className={`px-2 py-0.5 rounded-full text-xs font-medium border ${member.status === "active"
                                    ? "bg-accent-cyan/10 text-accent-cyan border-accent-cyan/20"
                                    : "bg-yellow-500/10 text-yellow-400 border-yellow-500/20"
                                    }`}>
                                    {member.status}
                                </span>
                                <div className="text-sm text-slate-300 w-20">{member.role}</div>
                                <button className="p-2 rounded hover:bg-white/10 text-slate-500 hover:text-white transition-colors">
                                    <MoreVertical className="w-4 h-4" />
                                </button>
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
}
