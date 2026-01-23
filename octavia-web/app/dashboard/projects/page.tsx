"use client";

import { motion, AnimatePresence } from "framer-motion";
import { Folder, Plus, Clock, CheckCircle, AlertCircle, Edit, Trash2, X, Save } from "lucide-react";
import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { useToast } from "@/hooks/use-toast";

interface Project {
    id: string;
    name: string;
    type: "Video Translation" | "Audio Translation" | "Subtitle Generation";
    status: "completed" | "in-progress" | "pending";
    date: string;
    files: number;
    description?: string;
    createdAt: string;
    updatedAt: string;
}

const projectTypes = [
    "Video Translation",
    "Audio Translation",
    "Subtitle Generation"
];

const initialProjects: Project[] = [
    {
        id: "1",
        name: "Marketing Video ES-FR",
        type: "Video Translation",
        status: "completed",
        date: "2024-11-20",
        files: 1,
        description: "Marketing video translation from English to French",
        createdAt: "2024-11-20T10:00:00Z",
        updatedAt: "2024-11-20T15:30:00Z"
    },
    {
        id: "2",
        name: "Podcast Series",
        type: "Audio Translation",
        status: "in-progress",
        date: "2024-11-22",
        files: 8,
        description: "Complete podcast series translation",
        createdAt: "2024-11-22T09:00:00Z",
        updatedAt: "2024-11-22T14:20:00Z"
    },
    {
        id: "3",
        name: "Tutorial Subtitles",
        type: "Subtitle Generation",
        status: "completed",
        date: "2024-11-18",
        files: 3,
        description: "Generate subtitles for tutorial videos",
        createdAt: "2024-11-18T11:00:00Z",
        updatedAt: "2024-11-18T16:45:00Z"
    },
    {
        id: "4",
        name: "Webinar Recording",
        type: "Video Translation",
        status: "pending",
        date: "2024-11-23",
        files: 1,
        description: "Translate recorded webinar session",
        createdAt: "2024-11-23T08:00:00Z",
        updatedAt: "2024-11-23T08:00:00Z"
    },
];

const getStatusBadge = (status: string) => {
    const statusMap: Record<string, { bg: string, text: string, icon: any }> = {
        "completed": { bg: "bg-green-500/10", text: "text-green-400", icon: CheckCircle },
        "in-progress": { bg: "bg-primary-purple/10", text: "text-primary-purple-bright", icon: Clock },
        "pending": { bg: "bg-orange-500/10", text: "text-orange-400", icon: AlertCircle }
    };
    return statusMap[status] || statusMap["pending"];
};

export default function ProjectsPage() {
    const router = useRouter();
    const { toast } = useToast();
    const [projects, setProjects] = useState<Project[]>([]);
    const [isModalOpen, setIsModalOpen] = useState(false);
    const [editingProject, setEditingProject] = useState<Project | null>(null);
    const [formData, setFormData] = useState({
        name: '',
        type: 'Video Translation' as Project['type'],
        status: 'pending' as Project['status'],
        description: '',
        files: 0
    });

    // Load projects from localStorage on component mount
    useEffect(() => {
        const storedProjects = localStorage.getItem('octavia_projects');
        if (storedProjects) {
            try {
                const parsedProjects = JSON.parse(storedProjects);
                setProjects(parsedProjects);
            } catch (error) {
                console.error('Failed to parse stored projects:', error);
                setProjects(initialProjects);
                localStorage.setItem('octavia_projects', JSON.stringify(initialProjects));
            }
        } else {
            setProjects(initialProjects);
            localStorage.setItem('octavia_projects', JSON.stringify(initialProjects));
        }
    }, []);

    // Save projects to localStorage whenever projects change
    useEffect(() => {
        if (projects.length > 0) {
            localStorage.setItem('octavia_projects', JSON.stringify(projects));
        }
    }, [projects]);

    const handleCreateProject = () => {
        setEditingProject(null);
        setFormData({
            name: '',
            type: 'Video Translation',
            status: 'pending',
            description: '',
            files: 0
        });
        setIsModalOpen(true);
    };

    const handleEditProject = (project: Project) => {
        setEditingProject(project);
        setFormData({
            name: project.name,
            type: project.type,
            status: project.status,
            description: project.description || '',
            files: project.files
        });
        setIsModalOpen(true);
    };

    const handleDeleteProject = (projectId: string) => {
        setProjects(prev => prev.filter(p => p.id !== projectId));
        toast({
            title: "Project deleted",
            description: "The project has been successfully deleted.",
        });
    };

    const handleSaveProject = () => {
        if (!formData.name.trim()) {
            toast({
                title: "Error",
                description: "Project name is required.",
                variant: "destructive",
            });
            return;
        }

        const now = new Date().toISOString();

        if (editingProject) {
            // Update existing project
            setProjects(prev => prev.map(p =>
                p.id === editingProject.id
                    ? {
                        ...p,
                        name: formData.name,
                        type: formData.type,
                        status: formData.status,
                        description: formData.description,
                        files: formData.files,
                        updatedAt: now
                    }
                    : p
            ));
            toast({
                title: "Project updated",
                description: "The project has been successfully updated.",
            });
        } else {
            // Create new project
            const newProject: Project = {
                id: Date.now().toString(),
                name: formData.name,
                type: formData.type,
                status: formData.status,
                description: formData.description,
                files: formData.files,
                date: new Date().toLocaleDateString('en-CA'), // YYYY-MM-DD format
                createdAt: now,
                updatedAt: now
            };
            setProjects(prev => [newProject, ...prev]);
            toast({
                title: "Project created",
                description: "The new project has been successfully created.",
            });
        }

        setIsModalOpen(false);
        setEditingProject(null);
    };

    const handleProjectClick = (project: Project) => {
        // Navigate to project details page
        router.push(`/dashboard/projects/${project.id}`);
    };

    return (
        <div className="space-y-8">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="font-display text-3xl font-black text-white text-glow-purple mb-2">Projects</h1>
                    <p className="text-slate-400 text-sm">Organize and manage your translation projects</p>
                </div>
                <button
                    onClick={handleCreateProject}
                    className="btn-border-beam group"
                >
                    <div className="btn-border-beam-inner flex items-center justify-center gap-2 py-2.5 px-5">
                        <Plus className="w-4 h-4" />
                        <span className="text-sm font-semibold">New Project</span>
                    </div>
                </button>
            </div>

            {/* Project Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <AnimatePresence>
                    {projects.map((project, index) => {
                        const statusConfig = getStatusBadge(project.status);
                        const StatusIcon = statusConfig.icon;

                        return (
                            <motion.div
                                key={project.id}
                                initial={{ opacity: 0, y: 20 }}
                                animate={{ opacity: 1, y: 0 }}
                                exit={{ opacity: 0, y: -20 }}
                                transition={{ delay: index * 0.1 }}
                                whileHover={{ y: -2 }}
                                onClick={() => handleProjectClick(project)}
                                className="glass-panel-glow p-5 cursor-pointer group relative"
                            >
                                <div className="glass-shine" />
                                <div className="relative z-10">
                                    <div className="flex items-start justify-between mb-3">
                                        <div className="flex items-center gap-3">
                                            <div className="flex items-center justify-center w-10 h-10 rounded-lg bg-primary-purple/10 border border-primary-purple/20">
                                                <Folder className="w-5 h-5 text-primary-purple-bright" />
                                            </div>
                                            <div className="flex-1">
                                                <h3 className="text-white font-bold text-base leading-tight">{project.name}</h3>
                                                <p className="text-slate-400 text-xs">{project.type}</p>
                                                {project.description && (
                                                    <p className="text-slate-500 text-xs mt-1 line-clamp-2">{project.description}</p>
                                                )}
                                            </div>
                                        </div>
                                        <div className="flex items-center gap-2">
                                            <div className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full ${statusConfig.bg} border border-white/10`}>
                                                <StatusIcon className={`w-3 h-3 ${statusConfig.text}`} />
                                                <span className={`text-xs font-semibold capitalize ${statusConfig.text}`}>{project.status.replace('-', ' ')}</span>
                                            </div>
                                            <div className="opacity-0 group-hover:opacity-100 transition-opacity flex gap-1">
                                                <button
                                                    onClick={(e) => {
                                                        e.stopPropagation();
                                                        handleEditProject(project);
                                                    }}
                                                    className="p-1.5 rounded-md bg-slate-700/50 hover:bg-slate-600/50 text-slate-300 hover:text-white transition-colors"
                                                >
                                                    <Edit className="w-3.5 h-3.5" />
                                                </button>
                                                <button
                                                    onClick={(e) => {
                                                        e.stopPropagation();
                                                        handleDeleteProject(project.id);
                                                    }}
                                                    className="p-1.5 rounded-md bg-red-700/50 hover:bg-red-600/50 text-red-300 hover:text-white transition-colors"
                                                >
                                                    <Trash2 className="w-3.5 h-3.5" />
                                                </button>
                                            </div>
                                        </div>
                                    </div>

                                    <div className="flex items-center gap-4 text-xs text-slate-500">
                                        <span>{project.files} file{project.files > 1 ? 's' : ''}</span>
                                        <span>•</span>
                                        <span>{project.date}</span>
                                    </div>
                                </div>
                            </motion.div>
                        );
                    })}
                </AnimatePresence>
            </div>

            {/* Empty State for New Users */}
            {projects.length === 0 && (
                <div className="glass-panel p-12 text-center">
                    <Folder className="w-16 h-16 text-slate-600 mx-auto mb-4" />
                    <h3 className="text-white text-lg font-bold mb-2">No projects yet</h3>
                    <p className="text-slate-400 text-sm mb-6">Create your first project to organize your translations</p>
                    <button
                        onClick={handleCreateProject}
                        className="btn-border-beam group"
                    >
                        <div className="btn-border-beam-inner flex items-center justify-center gap-2 py-2.5 px-5">
                            <Plus className="w-4 h-4" />
                            <span className="text-sm font-semibold">Create Project</span>
                        </div>
                    </button>
                </div>
            )}

            {/* Project Modal */}
            <AnimatePresence>
                {isModalOpen && (
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4"
                        onClick={() => setIsModalOpen(false)}
                    >
                        <motion.div
                            initial={{ scale: 0.9, opacity: 0 }}
                            animate={{ scale: 1, opacity: 1 }}
                            exit={{ scale: 0.9, opacity: 0 }}
                            onClick={(e) => e.stopPropagation()}
                            className="glass-panel max-w-sm w-full p-4"
                        >
                            <div className="flex items-center justify-between mb-3">
                                <h2 className="text-xl font-bold text-white">
                                    {editingProject ? 'Edit Project' : 'Create New Project'}
                                </h2>
                                <button
                                    onClick={() => setIsModalOpen(false)}
                                    className="p-1 rounded-md hover:bg-slate-700/50 text-slate-400 hover:text-white transition-colors"
                                >
                                    <X className="w-5 h-5" />
                                </button>
                            </div>

                            <div className="space-y-3">
                                <div>
                                    <label className="block text-sm font-medium text-slate-300 mb-2">
                                        Project Name *
                                    </label>
                                    <input
                                        type="text"
                                        value={formData.name}
                                        onChange={(e) => setFormData(prev => ({ ...prev, name: e.target.value }))}
                                        className="w-full px-3 py-2 bg-slate-800/50 border border-slate-600 rounded-md text-white placeholder-slate-400 focus:border-primary-purple focus:outline-none focus:ring-1 focus:ring-primary-purple"
                                        placeholder="Enter project name"
                                    />
                                </div>

                                <div>
                                    <label className="block text-sm font-medium text-slate-300 mb-1.5">
                                        Project Type
                                    </label>
                                    <select
                                        value={formData.type}
                                        onChange={(e) => setFormData(prev => ({ ...prev, type: e.target.value as Project['type'] }))}
                                        className="w-full px-3 py-2 bg-slate-800/50 border border-slate-600 rounded-md text-white focus:border-primary-purple focus:outline-none focus:ring-1 focus:ring-primary-purple"
                                    >
                                        {projectTypes.map(type => (
                                            <option key={type} value={type}>{type}</option>
                                        ))}
                                    </select>
                                </div>

                                <div>
                                    <label className="block text-sm font-medium text-slate-300 mb-1.5">
                                        Status
                                    </label>
                                    <select
                                        value={formData.status}
                                        onChange={(e) => setFormData(prev => ({ ...prev, status: e.target.value as Project['status'] }))}
                                        className="w-full px-3 py-2 bg-slate-800/50 border border-slate-600 rounded-md text-white focus:border-primary-purple focus:outline-none focus:ring-1 focus:ring-primary-purple"
                                    >
                                        <option value="pending">Pending</option>
                                        <option value="in-progress">In Progress</option>
                                        <option value="completed">Completed</option>
                                    </select>
                                </div>

                                <div>
                                    <label className="block text-sm font-medium text-slate-300 mb-1.5">
                                        Number of Files
                                    </label>
                                    <input
                                        type="number"
                                        min="0"
                                        value={formData.files}
                                        onChange={(e) => setFormData(prev => ({ ...prev, files: parseInt(e.target.value) || 0 }))}
                                        className="w-full px-3 py-2 bg-slate-800/50 border border-slate-600 rounded-md text-white focus:border-primary-purple focus:outline-none focus:ring-1 focus:ring-primary-purple"
                                    />
                                </div>

                                <div>
                                    <label className="block text-sm font-medium text-slate-300 mb-1.5">
                                        Description
                                    </label>
                                    <textarea
                                        value={formData.description}
                                        onChange={(e) => setFormData(prev => ({ ...prev, description: e.target.value }))}
                                        className="w-full px-3 py-2 bg-slate-800/50 border border-slate-600 rounded-md text-white placeholder-slate-400 focus:border-primary-purple focus:outline-none focus:ring-1 focus:ring-primary-purple resize-none"
                                        placeholder="Optional project description"
                                        rows={3}
                                    />
                                </div>
                            </div>

                            <div className="flex gap-3 mt-3">
                                <button
                                    onClick={() => setIsModalOpen(false)}
                                    className="flex-1 px-4 py-2 bg-slate-700 hover:bg-slate-600 text-white rounded-md transition-colors"
                                >
                                    Cancel
                                </button>
                                <button
                                    onClick={handleSaveProject}
                                    className="flex-1 btn-border-beam group"
                                >
                                    <div className="btn-border-beam-inner flex items-center justify-center gap-2 py-2">
                                        <Save className="w-4 h-4" />
                                        <span className="text-sm font-semibold">
                                            {editingProject ? 'Update' : 'Create'}
                                        </span>
                                    </div>
                                </button>
                            </div>
                        </motion.div>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
}
