"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import { Plus, Search, Folder, MoreVertical, Edit2, Trash2, Clock, CheckCircle, AlertCircle, Calendar } from "lucide-react";

interface Project {
    id: string;
    name: string;
    type: string;
    status: string;
    description: string;
    files: number;
    createdAt: string;
    updatedAt: string;
}

const mockProjects: Project[] = [
    {
        id: "1",
        name: "octavia test",
        type: "Video Translation",
        status: "pending",
        description: "",
        files: 3,
        createdAt: "2026-01-23T10:00:00Z",
        updatedAt: "2026-01-23T10:00:00Z"
    },
    {
        id: "2",
        name: "videos",
        type: "Video Translation",
        status: "in-progress",
        description: "",
        files: 1,
        createdAt: "2025-12-05T14:30:00Z",
        updatedAt: "2025-12-05T14:30:00Z"
    },
    {
        id: "3",
        name: "Podcast Series",
        type: "Audio Translation",
        status: "in-progress",
        description: "Complete podcast series translation",
        files: 8,
        createdAt: "2024-11-22T09:00:00Z",
        updatedAt: "2024-11-22T09:00:00Z"
    },
    {
        id: "4",
        name: "Tutorial Subtitles",
        type: "Subtitle Generation",
        status: "completed",
        description: "Generate subtitles for tutorial videos",
        files: 3,
        createdAt: "2024-11-18T11:20:00Z",
        updatedAt: "2024-11-18T11:20:00Z"
    },
    {
        id: "5",
        name: "Webinar Recording",
        type: "Video Translation",
        status: "pending",
        description: "Translate recorded webinar session",
        files: 1,
        createdAt: "2024-11-23T08:00:00Z",
        updatedAt: "2024-11-23T08:00:00Z"
    },
];

const getStatusBadge = (status: string) => {
    const statusMap: Record<string, { bg: string, text: string, icon: any }> = {
        "completed": { bg: "bg-accent-cyan/10", text: "text-accent-cyan", icon: CheckCircle },
        "in-progress": { bg: "bg-primary-purple/10", text: "text-primary-purple-bright", icon: Clock },
        "pending": { bg: "bg-orange-500/10", text: "text-orange-400", icon: AlertCircle }
    };
    return statusMap[status] || statusMap["pending"];
};

export default function ProjectsPage() {
    const router = useRouter();
    const [projects, setProjects] = useState<Project[]>([]);
    const [searchQuery, setSearchQuery] = useState("");
    const [isModalOpen, setIsModalOpen] = useState(false);
    const [editingProject, setEditingProject] = useState<Project | null>(null);
    const [formData, setFormData] = useState({
        name: "",
        description: "",
        type: "Video Translation"
    });

    // Load projects from localStorage on mount
    useEffect(() => {
        const storedProjects = localStorage.getItem('octavia_projects');
        if (storedProjects) {
            try {
                setProjects(JSON.parse(storedProjects));
            } catch (e) {
                console.error("Failed to parse projects", e);
                setProjects(mockProjects);
                localStorage.setItem('octavia_projects', JSON.stringify(mockProjects));
            }
        } else {
            // First time load, set mocks
            setProjects(mockProjects);
            localStorage.setItem('octavia_projects', JSON.stringify(mockProjects));
        }
    }, []);

    // Save projects to localStorage whenever they change
    const saveProjects = (newProjects: Project[]) => {
        setProjects(newProjects);
        localStorage.setItem('octavia_projects', JSON.stringify(newProjects));
    };

    const filteredProjects = projects.filter(p =>
        p.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        p.description.toLowerCase().includes(searchQuery.toLowerCase())
    );

    const handleCreateProject = () => {
        setEditingProject(null);
        setFormData({ name: "", description: "", type: "Video Translation" });
        setIsModalOpen(true);
    };

    const handleEditProject = (project: Project) => {
        setEditingProject(project);
        setFormData({
            name: project.name,
            description: project.description,
            type: project.type
        });
        setIsModalOpen(true);
    };

    const handleDeleteProject = (id: string) => {
        if (confirm("Are you sure you want to delete this project?")) {
            const updated = projects.filter(p => p.id !== id);
            saveProjects(updated);
        }
    };

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();

        if (editingProject) {
            const updated = projects.map(p =>
                p.id === editingProject.id
                    ? { ...p, ...formData, updatedAt: new Date().toISOString() }
                    : p
            );
            saveProjects(updated);
        } else {
            const newProject: Project = {
                id: Math.random().toString(36).substr(2, 9),
                ...formData,
                status: "pending",
                files: 0,
                createdAt: new Date().toISOString(),
                updatedAt: new Date().toISOString()
            };
            saveProjects([newProject, ...projects]);
        }

        setIsModalOpen(false);
    };

    return (
        <div className="space-y-8">
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="font-display text-4xl font-black text-white text-glow-purple">Projects</h1>
                    <p className="text-slate-400 text-sm mt-1">Organize and manage your translation projects</p>
                </div>
                <button
                    onClick={handleCreateProject}
                    className="flex items-center gap-2 px-6 py-3 bg-primary-purple hover:bg-primary-purple-bright text-white rounded-xl font-bold transition-all shadow-glow hover:scale-105 active:scale-95"
                >
                    <Plus className="w-5 h-5" />
                    New Project
                </button>
            </div>

            <div className="relative">
                <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-500" />
                <input
                    type="text"
                    placeholder="Search projects..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="w-full bg-white/5 border border-white/10 rounded-xl py-4 pl-12 pr-4 text-white placeholder:text-slate-600 focus:outline-none focus:border-primary-purple/50 focus:bg-white/10 transition-all"
                />
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                <AnimatePresence>
                    {filteredProjects.map((project, index) => {
                        const badge = getStatusBadge(project.status);
                        const BadgeIcon = badge.icon;

                        return (
                            <motion.div
                                key={project.id}
                                layout
                                initial={{ opacity: 0, y: 20 }}
                                animate={{ opacity: 1, y: 0 }}
                                exit={{ opacity: 0, scale: 0.95 }}
                                transition={{ delay: index * 0.05 }}
                                className="glass-panel p-6 group relative cursor-pointer hover:shadow-glow-purple transition-all border border-white/5 hover:border-primary-purple/30"
                                onClick={() => router.push(`/dashboard/projects/${project.id}`)}
                            >
                                <div className="absolute top-4 right-4 flex items-center gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                                    <button
                                        onClick={(e) => {
                                            e.stopPropagation();
                                            handleEditProject(project);
                                        }}
                                        className="p-2 bg-white/5 hover:bg-white/10 rounded-lg text-slate-400 hover:text-white transition-colors"
                                    >
                                        <Edit2 className="w-4 h-4" />
                                    </button>
                                    <button
                                        onClick={(e) => {
                                            e.stopPropagation();
                                            handleDeleteProject(project.id);
                                        }}
                                        className="p-2 bg-red-500/10 hover:bg-red-500/20 rounded-lg text-red-400 hover:text-red-300 transition-colors"
                                    >
                                        <Trash2 className="w-4 h-4" />
                                    </button>
                                </div>

                                <div className="flex items-start gap-4 mb-6">
                                    <div className="w-12 h-12 rounded-xl bg-primary-purple/10 flex items-center justify-center text-primary-purple-bright group-hover:scale-110 transition-transform">
                                        <Folder className="w-6 h-6" />
                                    </div>
                                    <div className="flex-1 pr-14">
                                        <h3 className="text-white font-bold text-lg leading-tight mb-1">{project.name}</h3>
                                        <p className="text-slate-500 text-xs">{project.type}</p>
                                    </div>
                                </div>

                                {project.description && (
                                    <p className="text-slate-400 text-sm mb-6 line-clamp-2">{project.description}</p>
                                )}

                                <div className="flex items-center justify-between mt-auto">
                                    <div className="flex items-center gap-4 text-xs text-slate-500">
                                        <div className="flex items-center gap-1">
                                            {project.files} files
                                        </div>
                                        <div className="flex items-center gap-1">
                                            <Calendar className="w-3 h-3" />
                                            {new Date(project.createdAt).toLocaleDateString()}
                                        </div>
                                    </div>

                                    <div className={`flex items-center gap-1 px-3 py-1 rounded-full text-[10px] font-bold uppercase tracking-wider ${badge.bg} ${badge.text}`}>
                                        <BadgeIcon className="w-3 h-3" />
                                        {project.status.replace('-', ' ')}
                                    </div>
                                </div>
                            </motion.div>
                        );
                    })}
                </AnimatePresence>
            </div>

            {/* Modal for Create/Edit */}
            {isModalOpen && (
                <div className="fixed inset-0 z-50 flex items-center justify-center p-6 bg-black/60 backdrop-blur-sm">
                    <motion.div
                        initial={{ opacity: 0, scale: 0.95 }}
                        animate={{ opacity: 1, scale: 1 }}
                        className="glass-panel w-full max-w-lg p-8 relative"
                    >
                        <button
                            onClick={() => setIsModalOpen(false)}
                            className="absolute top-6 right-6 text-slate-500 hover:text-white transition-colors"
                        >
                            <Trash2 className="w-6 h-6" />
                        </button>

                        <h2 className="text-2xl font-bold text-white mb-6">
                            {editingProject ? "Edit Project" : "Create New Project"}
                        </h2>

                        <form onSubmit={handleSubmit} className="space-y-6">
                            <div>
                                <label className="block text-slate-400 text-sm mb-2">Project Name</label>
                                <input
                                    type="text"
                                    required
                                    value={formData.name}
                                    onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                                    className="w-full bg-white/5 border border-white/10 rounded-xl py-3 px-4 text-white focus:outline-none focus:border-primary-purple/50 transition-all"
                                    placeholder="e.g. My Awesome Video"
                                />
                            </div>

                            <div>
                                <label className="block text-slate-400 text-sm mb-2">Project Type</label>
                                <select
                                    value={formData.type}
                                    onChange={(e) => setFormData({ ...formData, type: e.target.value })}
                                    className="w-full bg-white/5 border border-white/10 rounded-xl py-3 px-4 text-white focus:outline-none focus:border-primary-purple/50 transition-all"
                                >
                                    <option value="Video Translation">Video Translation</option>
                                    <option value="Audio Translation">Audio Translation</option>
                                    <option value="Subtitle Generation">Subtitle Generation</option>
                                </select>
                            </div>

                            <div>
                                <label className="block text-slate-400 text-sm mb-2">Description (Optional)</label>
                                <textarea
                                    value={formData.description}
                                    onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                                    className="w-full bg-white/5 border border-white/10 rounded-xl py-3 px-4 text-white focus:outline-none focus:border-primary-purple/50 transition-all min-h-[100px]"
                                    placeholder="Tell us what this project is about..."
                                />
                            </div>

                            <button
                                type="submit"
                                className="w-full py-4 bg-primary-purple hover:bg-primary-purple-bright text-white rounded-xl font-bold transition-all shadow-glow"
                            >
                                {editingProject ? "Update Project" : "Create Project"}
                            </button>
                        </form>
                    </motion.div>
                </div>
            )}
        </div>
    );
}
