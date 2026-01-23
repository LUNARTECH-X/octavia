"use client";

import { motion, AnimatePresence } from "framer-motion";
import { Folder, Clock, CheckCircle, AlertCircle, X, Upload, Download, FileVideo, FileAudio, FileText, ArrowLeft, Play, Edit, Trash2 } from "lucide-react";
import { useState, useEffect, useRef } from "react";
import { useParams, useRouter } from "next/navigation";
import { useToast } from "@/hooks/use-toast";
import { useUser } from "@/contexts/UserContext";
import Link from "next/link";

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

type JobType = "Video Translation" | "Audio Translation" | "Subtitle Generation" | "Subtitle Translation" | "Subtitle to Audio";

interface ProjectFile {
    id: string;
    name: string;
    type: string;
    size: number;
    uploadedAt: string;
    status: "uploaded" | "processing" | "completed" | "failed";
    jobType?: JobType;
    jobId?: string;
    downloadUrl?: string;
}

interface Job {
    id: string;
    type: string;
    status: string;
    progress: number;
    created_at: string;
    download_url?: string;
}

const getStatusBadge = (status: string) => {
    const statusMap: Record<string, { bg: string, text: string, icon: any }> = {
        "completed": { bg: "bg-green-500/10", text: "text-green-400", icon: CheckCircle },
        "in-progress": { bg: "bg-primary-purple/10", text: "text-primary-purple-bright", icon: Clock },
        "pending": { bg: "bg-orange-500/10", text: "text-orange-400", icon: AlertCircle },
        "uploaded": { bg: "bg-blue-500/10", text: "text-blue-400", icon: CheckCircle },
        "processing": { bg: "bg-primary-purple/10", text: "text-primary-purple-bright", icon: Clock },
        "failed": { bg: "bg-red-500/10", text: "text-red-400", icon: AlertCircle }
    };
    return statusMap[status] || statusMap["pending"];
};

const getFileIcon = (type: string) => {
    if (type.includes('video')) return FileVideo;
    if (type.includes('audio')) return FileAudio;
    return FileText;
};

const languageNames: Record<string, string> = {
    'en': 'English',
    'es': 'Spanish',
    'fr': 'French',
    'de': 'German',
    'it': 'Italian',
    'pt': 'Portuguese',
    'ru': 'Russian',
    'ja': 'Japanese',
    'ko': 'Korean',
    'zh-cn': 'Chinese (Mandarin)',
    'nl': 'Dutch',
    'pl': 'Polish',
    'tr': 'Turkish',
    'sv': 'Swedish'
};

export default function ProjectDetailPage() {
    const params = useParams();
    const router = useRouter();
    const { toast } = useToast();
    const { user } = useUser();
    const [project, setProject] = useState<Project | null>(null);
    const [files, setFiles] = useState<ProjectFile[]>([]);
    const [jobs, setJobs] = useState<Job[]>([]);
    const [loading, setLoading] = useState(true);

    const [showUploadModal, setShowUploadModal] = useState(false);
    const [selectedJobType, setSelectedJobType] = useState<JobType>('Video Translation');

    const [editingFile, setEditingFile] = useState<ProjectFile | null>(null);
    const [showEditModal, setShowEditModal] = useState(false);
    const [showDeleteModal, setShowDeleteModal] = useState(false);
    const [fileToDelete, setFileToDelete] = useState<ProjectFile | null>(null);

    const fileInputRef = useRef<HTMLInputElement>(null);



    const getToken = (): string | null => {
        if (typeof window === 'undefined') return null;
        const userStr = localStorage.getItem('octavia_user');
        if (userStr) {
            try {
                const user = JSON.parse(userStr);
                return user.token || null;
            } catch (error) {
                console.error('Failed to parse user token:', error);
                return null;
            }
        }
        return null;
    };

    const projectId = params.id as string;

    // Load project data
    useEffect(() => {
        loadProject();
    }, [projectId, router, toast]);

    // Reload files when component gains focus (when navigating back)
    useEffect(() => {
        const handleVisibilityChange = () => {
            if (!document.hidden && project) {
                loadProjectFiles(project.id);
            }
        };

        const handleFocus = () => {
            if (project) {
                loadProjectFiles(project.id);
            }
        };

        document.addEventListener('visibilitychange', handleVisibilityChange);
        window.addEventListener('focus', handleFocus);

        return () => {
            document.removeEventListener('visibilitychange', handleVisibilityChange);
            window.removeEventListener('focus', handleFocus);
        };
    }, [project]);

    // Listen for storage changes from other tabs
    useEffect(() => {
        const handleStorageChange = (e: StorageEvent) => {
            if (e.key === `octavia_project_files_${projectId}` && project) {
                loadProjectFiles(project.id);
            }
        };

        window.addEventListener('storage', handleStorageChange);
        return () => window.removeEventListener('storage', handleStorageChange);
    }, [projectId, project]);

    const loadProjectFiles = (projectId: string) => {
        const storedFiles = localStorage.getItem(`octavia_project_files_${projectId}`);
        if (storedFiles) {
            try {
                const files = JSON.parse(storedFiles);
                setFiles(files);
            } catch (error) {
                console.error('Failed to parse project files:', error);
                setFiles([]);
            }
        } else {
            setFiles([]);
        }

        const storedJobs = localStorage.getItem(`octavia_project_jobs_${projectId}`);
        if (storedJobs) {
            try {
                const jobs = JSON.parse(storedJobs);
                setJobs(jobs);
            } catch (error) {
                console.error('Failed to parse project jobs:', error);
                setJobs([]);
            }
        } else {
            setJobs([]);
        }
    };

    const loadProject = () => {
        const storedProjects = localStorage.getItem('octavia_projects');
        if (storedProjects) {
            try {
                const projects = JSON.parse(storedProjects);
                const foundProject = projects.find((p: Project) => p.id === projectId);
                if (foundProject) {
                    setProject(foundProject);
                    // Load associated files and jobs from localStorage
                    loadProjectFiles(projectId);
                } else {
                    toast({
                        title: "Project not found",
                        description: "The requested project could not be found.",
                        variant: "destructive",
                    });
                    router.push('/dashboard/projects');
                }
            } catch (error) {
                console.error('Failed to parse projects:', error);
                router.push('/dashboard/projects');
            }
        } else {
            router.push('/dashboard/projects');
        }
        setLoading(false);
    };

    const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
        if (!e.target.files) return;

        const file = e.target.files[0];

        // Close modal and redirect immediately
        setShowUploadModal(false);
        await redirectToJobPage(file, selectedJobType);
    };

    const redirectToJobPage = async (file: File, jobType: JobType) => {
        if (!project) return;

        // Use the actual file name
        const displayName = file.name;

        // Store the file as a blob URL for the job page to access
        const fileBlob = new Blob([file], { type: file.type });
        const fileUrl = URL.createObjectURL(fileBlob);

        // Store project context in localStorage for the job page to pick up
        localStorage.setItem('octavia_project_context', JSON.stringify({
            projectId: project.id,
            fileName: file.name,
            fileType: file.type,
            fileSize: file.size,
            fileUrl: fileUrl,
            projectType: jobType
        }));

        // For demo accounts, don't store in project, just redirect
        if (user?.email === "demo@octavia.com") {
            redirectToJobType(jobType);
            return;
        }

        // For non-demo accounts, store the file metadata in the project
        const newFile: ProjectFile = {
            id: Date.now().toString(),
            name: displayName,
            type: file.type,
            size: file.size,
            uploadedAt: new Date().toISOString(),
            status: "uploaded",
            jobType: jobType
        };

        // Save file to localStorage
        const currentFiles = JSON.parse(localStorage.getItem(`octavia_project_files_${project.id}`) || '[]');
        const updatedFiles = [...currentFiles, newFile];
        localStorage.setItem(`octavia_project_files_${project.id}`, JSON.stringify(updatedFiles));
        
        setFiles(prev => [...prev, newFile]);
        updateProjectFileCount(project.id, updatedFiles.length);

        redirectToJobType(jobType);
    };

    const startTranslationFromFile = (file: ProjectFile) => {
        if (!project) return;

        // Check if there's existing context with file data
        const existingContext = localStorage.getItem('octavia_project_context');
        let context = null;

        if (existingContext) {
            try {
                context = JSON.parse(existingContext);
                // If existing context has fileUrl, keep it and just update project info
                if (context.fileUrl) {
                    context.projectId = project.id;
                    context.displayName = file.name;
                    localStorage.setItem('octavia_project_context', JSON.stringify(context));
                    redirectToJobType(project.type);
                    return;
                }
            } catch (error) {
                console.error('Failed to parse existing context:', error);
            }
        }

        // No existing context with fileUrl, create new context
        localStorage.setItem('octavia_project_context', JSON.stringify({
            projectId: project.id,
            fileId: file.id,
            fileName: file.name,
            fileType: file.type,
            fileSize: file.size,
            downloadUrl: file.downloadUrl
        }));

        redirectToJobType(project.type);
    };

    const openDeleteModal = (file: ProjectFile) => {
        setFileToDelete(file);
        setShowDeleteModal(true);
    };

    const confirmDeleteFile = () => {
        if (!project || !fileToDelete) return;

        // Remove file from localStorage
        const currentFiles = JSON.parse(localStorage.getItem(`octavia_project_files_${project.id}`) || '[]');
        const updatedFiles = currentFiles.filter((f: ProjectFile) => f.id !== fileToDelete.id);
        localStorage.setItem(`octavia_project_files_${project.id}`, JSON.stringify(updatedFiles));

        // Update state
        setFiles(updatedFiles);
        updateProjectFileCount(project.id, updatedFiles.length);

        // Close modal and reset state
        setShowDeleteModal(false);
        setFileToDelete(null);

        toast({
            title: "File deleted",
            description: "The file has been removed from the project.",
        });
    };

    const openEditModal = (file: ProjectFile) => {
        setEditingFile(file);
        setShowEditModal(true);
    };

    const updateFile = () => {
        if (!editingFile || !project) return;

        // Update file in localStorage
        const currentFiles = JSON.parse(localStorage.getItem(`octavia_project_files_${project.id}`) || '[]');
        const updatedFiles = currentFiles.map((f: ProjectFile) => 
            f.id === editingFile.id ? editingFile : f
        );
        localStorage.setItem(`octavia_project_files_${project.id}`, JSON.stringify(updatedFiles));
        
        // Update state
        setFiles(updatedFiles);
        setShowEditModal(false);
        setEditingFile(null);
        
        toast({
            title: "File updated",
            description: "The file has been updated successfully.",
        });
    };

    const redirectToJobType = (jobType: JobType) => {
        switch (jobType) {
            case "Video Translation":
                router.push('/dashboard/video');
                break;
            case "Audio Translation":
                router.push('/dashboard/audio');
                break;
            case "Subtitle Generation":
                router.push('/dashboard/subtitles');
                break;
            case "Subtitle Translation":
                router.push('/dashboard/subtitles/translate');
                break;
            case "Subtitle to Audio":
                router.push('/dashboard/audio/subtitle-to-audio');
                break;
            default:
                router.push('/dashboard/video');
        }
    };

    const getValidFileTypes = (jobType: JobType): string[] => {
        switch (jobType) {
            case "Video Translation":
                return ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.wmv', '.flv'];
            case "Audio Translation":
                return ['.mp3', '.wav', '.m4a', '.flac', '.aac', '.ogg'];
            case "Subtitle Generation":
                return ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.mp3', '.wav', '.m4a'];
            case "Subtitle Translation":
                return ['.srt', '.vtt', '.ass', '.ssa'];
            case "Subtitle to Audio":
                return ['.srt', '.vtt', '.ass', '.ssa'];
            default:
                return [];
        }
    };

    const updateProjectFileCount = (projectId: string, count: number) => {
        const storedProjects = localStorage.getItem('octavia_projects');
        if (storedProjects) {
            try {
                const projects = JSON.parse(storedProjects);
                const updatedProjects = projects.map((p: Project) =>
                    p.id === projectId ? { ...p, files: count, updatedAt: new Date().toISOString() } : p
                );
                localStorage.setItem('octavia_projects', JSON.stringify(updatedProjects));
                setProject(prev => prev ? { ...prev, files: count } : null);
            } catch (error) {
                console.error('Failed to update project:', error);
            }
        }
    };



    const getProgressRoute = (projectType: string): string => {
        switch (projectType) {
            case "Video Translation":
                return "video/progress";
            case "Audio Translation":
                return "audio/progress";
            case "Subtitle Generation":
                return "subtitles/progress";
            default:
                return "video/progress";
        }
    };

    if (loading) {
        return (
            <div className="min-h-screen w-full bg-bg-dark flex items-center justify-center">
                <div className="text-center">
                    <div className="w-16 h-16 mx-auto mb-4 border-4 border-primary-purple/30 border-t-primary-purple rounded-full animate-spin" />
                    <p className="text-slate-400">Loading project...</p>
                </div>
            </div>
        );
    }

    if (!project) {
        return (
            <div className="min-h-screen w-full bg-bg-dark flex items-center justify-center">
                <div className="text-center">
                    <h1 className="text-2xl font-bold text-white mb-4">Project Not Found</h1>
                    <Link href="/dashboard/projects" className="btn-border-beam group">
                        <div className="btn-border-beam-inner flex items-center justify-center gap-2 py-2 px-4">
                            <ArrowLeft className="w-4 h-4" />
                            <span>Back to Projects</span>
                        </div>
                    </Link>
                </div>
            </div>
        );
    }

    const statusConfig = getStatusBadge(project.status);
    const StatusIcon = statusConfig.icon;

    return (
        <div className="space-y-8">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div className="flex items-center gap-4">
                    <Link href="/dashboard/projects" className="p-2 rounded-md hover:bg-slate-700/50 transition-colors">
                        <ArrowLeft className="w-5 h-5 text-slate-400" />
                    </Link>
                    <div>
                        <div className="flex items-center gap-3 mb-1">
                            <h1 className="font-display text-3xl font-black text-white text-glow-purple">{project.name}</h1>
                            <div className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full ${statusConfig.bg} border border-white/10`}>
                                <StatusIcon className={`w-3 h-3 ${statusConfig.text}`} />
                                <span className={`text-xs font-semibold capitalize ${statusConfig.text}`}>{project.status.replace('-', ' ')}</span>
                            </div>
                        </div>
                        <p className="text-slate-400 text-sm">{project.type} • {project.files} files • Created {new Date(project.createdAt).toLocaleDateString()}</p>
                        {project.description && (
                            <p className="text-slate-500 text-sm mt-1">{project.description}</p>
                        )}
                    </div>
                </div>
                <button
                    onClick={() => setShowUploadModal(true)}
                    className="btn-border-beam group"
                >
                    <div className="btn-border-beam-inner flex items-center justify-center gap-2 py-2.5 px-5">
                        <Upload className="w-4 h-4" />
                        <span className="text-sm font-semibold">Upload File</span>
                    </div>
                </button>
            </div>

            {/* Files Section */}
            <div className="space-y-4">
                <h2 className="text-xl font-bold text-white">Project Files</h2>
                
                {files.length === 0 ? (
                    <div className="glass-panel p-12 text-center">
                        <Folder className="w-16 h-16 text-slate-600 mx-auto mb-4" />
                        <h3 className="text-white text-lg font-bold mb-2">No files yet</h3>
                        <p className="text-slate-400 text-sm mb-6">Upload files to start translating</p>
                        <button
                            onClick={() => setShowUploadModal(true)}
                            className="btn-border-beam group"
                        >
                            <div className="btn-border-beam-inner flex items-center justify-center gap-2 py-2.5 px-5">
                                <Upload className="w-4 h-4" />
                                <span className="text-sm font-semibold">Upload File</span>
                            </div>
                        </button>
                    </div>
                ) : (
                    <div className="grid grid-cols-1 gap-4">
                        {files.map((file, index) => {
                            const fileStatusConfig = getStatusBadge(file.status);
                            const FileIcon = getFileIcon(file.type);
                            const FileStatusIcon = fileStatusConfig.icon;

                            return (
                                <motion.div
                                    key={file.id}
                                    initial={{ opacity: 0, y: 20 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    transition={{ delay: index * 0.1 }}
                                    className="glass-panel p-4"
                                >
                                    <div className="flex items-center justify-between">
                                        <div className="flex items-center gap-3">
                                            <FileIcon className="w-8 h-8 text-slate-400" />
                                            <div className="flex-1">
                                                <h3 className="text-white font-semibold">{file.name}</h3>

                                                <div className="flex flex-wrap gap-2 text-xs text-slate-400 mt-1">
                                                    <span>{(file.size / (1024 * 1024)).toFixed(2)} MB</span>
                                                    <span>•</span>
                                                    <span>{file.type || 'Unknown type'}</span>
                                                    {file.jobType && (
                                                        <>
                                                            <span>•</span>
                                                            <span>{file.jobType}</span>
                                                        </>
                                                    )}
                                                </div>
                                                <p className="text-slate-500 text-xs mt-1">
                                                    Uploaded {new Date(file.uploadedAt).toLocaleDateString()} at {new Date(file.uploadedAt).toLocaleTimeString()}
                                                </p>
                                            </div>
                                        </div>
                                        <div className="flex items-center gap-3">
                                            <div className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full ${fileStatusConfig.bg} border border-white/10`}>
                                                <FileStatusIcon className={`w-3 h-3 ${fileStatusConfig.text}`} />
                                                <span className={`text-xs font-semibold capitalize ${fileStatusConfig.text}`}>{file.status}</span>
                                            </div>
                                            {file.status === "uploaded" && (
                                                <button
                                                    onClick={() => startTranslationFromFile(file)}
                                                    className="btn-border-beam group"
                                                >
                                                    <div className="btn-border-beam-inner flex items-center justify-center gap-2 py-2 px-4">
                                                        <Play className="w-4 h-4" />
                                                        <span className="text-sm font-semibold">Continue Translation</span>
                                                    </div>
                                                </button>
                                            )}
                                            {file.status === "processing" && (
                                                <button
                                                    onClick={() => file.jobId && router.push(`/dashboard/${getProgressRoute(project.type)}?jobId=${file.jobId}`)}
                                                    className="btn-border-beam group"
                                                >
                                                    <div className="btn-border-beam-inner flex items-center justify-center gap-2 py-2 px-4">
                                                        <Clock className="w-4 h-4" />
                                                        <span className="text-sm font-semibold">View Progress</span>
                                                    </div>
                                                </button>
                                            )}
                                            {file.downloadUrl && (
                                                <button className="p-2 rounded-md bg-green-700/50 hover:bg-green-600/50 text-green-300 hover:text-white transition-colors">
                                                    <Download className="w-4 h-4" />
                                                </button>
                                            )}
                                            <button 
                                                onClick={() => openEditModal(file)}
                                                className="p-2 rounded-md bg-blue-700/50 hover:bg-blue-600/50 text-blue-300 hover:text-white transition-colors"
                                            >
                                                <Edit className="w-4 h-4" />
                                            </button>
                                            <button
                                                onClick={() => openDeleteModal(file)}
                                                className="p-2 rounded-md bg-red-700/50 hover:bg-red-600/50 text-red-300 hover:text-white transition-colors"
                                            >
                                                <Trash2 className="w-4 h-4" />
                                            </button>
                                        </div>
                                    </div>
                                </motion.div>
                            );
                        })}
                    </div>
                )}
            </div>

            {/* Jobs Section */}
            {jobs.length > 0 && (
                <div className="space-y-4">
                    <h2 className="text-xl font-bold text-white">Translation Jobs</h2>
                    <div className="grid grid-cols-1 gap-4">
                        {jobs.map((job, index) => {
                            const jobStatusConfig = getStatusBadge(job.status);
                            const JobStatusIcon = jobStatusConfig.icon;

                            return (
                                <motion.div
                                    key={job.id}
                                    initial={{ opacity: 0, y: 20 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    transition={{ delay: index * 0.1 }}
                                    className="glass-panel p-4"
                                >
                                    <div className="flex items-center justify-between">
                                        <div>
                                            <h3 className="text-white font-semibold">{job.type} Job</h3>
                                            <p className="text-slate-400 text-sm">
                                                Started {new Date(job.created_at).toLocaleDateString()} • 
                                                {job.progress}% complete
                                            </p>
                                        </div>
                                        <div className="flex items-center gap-3">
                                            <div className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full ${jobStatusConfig.bg} border border-white/10`}>
                                                <JobStatusIcon className={`w-3 h-3 ${jobStatusConfig.text}`} />
                                                <span className={`text-xs font-semibold capitalize ${jobStatusConfig.text}`}>{job.status}</span>
                                            </div>
                                            {job.download_url && (
                                                <button className="p-2 rounded-md bg-green-700/50 hover:bg-green-600/50 text-green-300 hover:text-white transition-colors">
                                                    <Download className="w-4 h-4" />
                                                </button>
                                            )}
                                        </div>
                                    </div>
                                    {job.status === "processing" && (
                                        <div className="mt-4">
                                            <div className="w-full bg-slate-700 rounded-full h-2">
                                                <div 
                                                    className="bg-primary-purple-bright h-2 rounded-full transition-all duration-300" 
                                                    style={{ width: `${job.progress}%` }}
                                                />
                                            </div>
                                        </div>
                                    )}
                                </motion.div>
                            );
                        })}
                    </div>
                </div>
            )}

            {/* Upload Modal */}
            <AnimatePresence>
                {showUploadModal && (
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4"
                        onClick={() => setShowUploadModal(false)}
                    >
                        <motion.div
                            initial={{ scale: 0.9, opacity: 0 }}
                            animate={{ scale: 1, opacity: 1 }}
                            exit={{ scale: 0.9, opacity: 0 }}
                            onClick={(e) => e.stopPropagation()}
                            className="glass-panel max-w-md w-full p-6"
                        >
                             <div className="flex items-center justify-between mb-6">
                                 <h2 className="text-xl font-bold text-white">Start New Job</h2>
                                <button
                                    onClick={() => setShowUploadModal(false)}
                                    className="p-1 rounded-md hover:bg-slate-700/50 text-slate-400 hover:text-white transition-colors"
                                >
                                    <X className="w-5 h-5" />
                                </button>
                            </div>

                             <div className="space-y-4">
                                 <div>
                                     <label className="block text-sm font-medium text-slate-300 mb-2">
                                         Job Type
                                     </label>
                                     <select
                                         value={selectedJobType}
                                         onChange={(e) => setSelectedJobType(e.target.value as JobType)}
                                         className="w-full px-3 py-2 bg-slate-800/50 border border-slate-600 rounded-md text-white focus:border-primary-purple focus:outline-none focus:ring-1 focus:ring-primary-purple"
                                     >
                                         <option value="Video Translation">Video Translation</option>
                                         <option value="Audio Translation">Audio Translation</option>
                                         <option value="Subtitle Generation">Subtitle Generation</option>
                                         <option value="Subtitle Translation">Subtitle Translation</option>
                                         <option value="Subtitle to Audio">Subtitle to Audio</option>
                                     </select>
                                 </div>



                                 <div>
                                     <label className="block text-sm font-medium text-slate-300 mb-2">
                                         Select File
                                     </label>
                                     <input
                                         ref={fileInputRef}
                                         type="file"
                                         onChange={handleFileUpload}
                                         className="w-full px-3 py-2 bg-slate-800/50 border border-slate-600 rounded-md text-white file:mr-4 file:py-1 file:px-3 file:rounded-md file:border-0 file:text-sm file:font-semibold file:bg-primary-purple file:text-white hover:file:bg-primary-purple-bright"
                                     />
                                 </div>


                             </div>
                        </motion.div>
                    </motion.div>
                )}
             </AnimatePresence>

             {/* Edit File Modal */}
             <AnimatePresence>
                 {showEditModal && editingFile && (
                     <motion.div
                         initial={{ opacity: 0 }}
                         animate={{ opacity: 1 }}
                         exit={{ opacity: 0 }}
                         className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4"
                         onClick={() => setShowEditModal(false)}
                     >
                         <motion.div
                             initial={{ scale: 0.9, opacity: 0 }}
                             animate={{ scale: 1, opacity: 1 }}
                             exit={{ scale: 0.9, opacity: 0 }}
                             onClick={(e) => e.stopPropagation()}
                             className="glass-panel max-w-md w-full p-6"
                         >
                             <div className="flex items-center justify-between mb-6">
                                 <h2 className="text-xl font-bold text-white">Edit File</h2>
                                 <button
                                     onClick={() => setShowEditModal(false)}
                                     className="p-1 rounded-md hover:bg-slate-700/50 text-slate-400 hover:text-white transition-colors"
                                 >
                                     <X className="w-5 h-5" />
                                 </button>
                             </div>

                              <div className="space-y-4">
                                  <div className="p-4 bg-slate-800/50 rounded-md">
                                      <p className="text-slate-300 text-sm">
                                          <strong>File:</strong> {editingFile.name}
                                      </p>
                                      <p className="text-slate-400 text-xs mt-1">
                                          Filename cannot be changed - using the original uploaded filename.
                                      </p>
                                  </div>

                                  <div>
                                      <label className="block text-sm font-medium text-slate-300 mb-2">
                                          Job Type
                                      </label>
                                      <select
                                          value={editingFile.jobType || 'Video Translation'}
                                          onChange={(e) => setEditingFile({ ...editingFile, jobType: e.target.value as JobType })}
                                          className="w-full px-3 py-2 bg-slate-800/50 border border-slate-600 rounded-md text-white focus:border-primary-purple focus:outline-none focus:ring-1 focus:ring-primary-purple"
                                      >
                                          <option value="Video Translation">Video Translation</option>
                                          <option value="Audio Translation">Audio Translation</option>
                                          <option value="Subtitle Generation">Subtitle Generation</option>
                                          <option value="Subtitle Translation">Subtitle Translation</option>
                                          <option value="Subtitle to Audio">Subtitle to Audio</option>
                                      </select>
                                  </div>

                                 <div className="flex gap-3 pt-4">
                                     <button
                                         onClick={() => setShowEditModal(false)}
                                         className="flex-1 px-4 py-2 bg-slate-700/50 hover:bg-slate-600/50 text-slate-300 hover:text-white rounded-md transition-colors"
                                     >
                                         Cancel
                                     </button>
                                     <button
                                         onClick={updateFile}
                                         className="flex-1 px-4 py-2 bg-primary-purple hover:bg-primary-purple-bright text-white rounded-md transition-colors"
                                     >
                                         Update File
                                     </button>
                                 </div>
                             </div>
                         </motion.div>
                     </motion.div>
                 )}
             </AnimatePresence>

             {/* Delete Confirmation Modal */}
             <AnimatePresence>
                 {showDeleteModal && fileToDelete && (
                     <motion.div
                         initial={{ opacity: 0 }}
                         animate={{ opacity: 1 }}
                         exit={{ opacity: 0 }}
                         className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4"
                         onClick={() => setShowDeleteModal(false)}
                     >
                         <motion.div
                             initial={{ scale: 0.9, opacity: 0 }}
                             animate={{ scale: 1, opacity: 1 }}
                             exit={{ scale: 0.9, opacity: 0 }}
                             onClick={(e) => e.stopPropagation()}
                             className="glass-panel max-w-sm w-full p-6"
                         >
                             <div className="flex items-center justify-between mb-4">
                                 <h2 className="text-xl font-bold text-white">Delete File</h2>
                                 <button
                                     onClick={() => setShowDeleteModal(false)}
                                     className="p-1 rounded-md hover:bg-slate-700/50 text-slate-400 hover:text-white transition-colors"
                                 >
                                     <X className="w-5 h-5" />
                                 </button>
                             </div>

                             <div className="mb-6">
                                 <p className="text-slate-300 text-sm">
                                     Are you sure you want to delete this file?
                                 </p>
                                 <p className="text-white font-semibold mt-2">
                                     "{fileToDelete.name}"
                                 </p>
                                 <p className="text-slate-400 text-xs mt-1">
                                     This action cannot be undone.
                                 </p>
                             </div>

                             <div className="flex gap-3">
                                 <button
                                     onClick={() => setShowDeleteModal(false)}
                                     className="flex-1 px-4 py-2 bg-slate-700/50 hover:bg-slate-600/50 text-slate-300 hover:text-white rounded-md transition-colors"
                                 >
                                     Cancel
                                 </button>
                                 <button
                                     onClick={confirmDeleteFile}
                                     className="flex-1 px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-md transition-colors"
                                 >
                                     Delete File
                                 </button>
                             </div>
                         </motion.div>
                     </motion.div>
                 )}
             </AnimatePresence>
        </div>
    );
}