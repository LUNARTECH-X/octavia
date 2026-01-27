# Octavia Projects Page - Comprehensive Documentation

## Overview

The Octavia Projects Page serves as a centralized hub for managing translation projects, allowing users to create projects, upload files, track progress, and seamlessly transition to specific job processing pages. This documentation covers the complete evolution of the projects page functionality, integration with job pages, technical implementations, challenges faced, and solutions implemented.

## Architecture Overview

### Core Components
- **Project Management**: Create, view, and organize translation projects
- **File Management**: Upload, edit, delete, and organize project files
- **Job Type Integration**: Seamless transitions to specialized job processing pages
- **State Persistence**: Local storage-based data management for demo functionality

### Integration Points
- **Job Pages**: Video, Audio, Subtitles, Subtitle Translation, Subtitle-to-Audio
- **File Storage**: Local storage with blob URL management
- **Progress Tracking**: Real-time job status updates
- **User Context**: Authentication and user-specific project isolation

## Feature Evolution Timeline

### Phase 1: Basic Project Creation and File Upload

#### Initial Implementation
- **Basic project listing** with create project functionality
- **Simple file upload** directly to job pages
- **Local storage** for project metadata
- **Direct routing** to job-specific pages

#### Technical Approach
- Used Next.js dynamic routing (`[id]`) for project-specific pages
- Implemented localStorage for data persistence in demo environment
- Basic form validation and error handling

#### Challenges Encountered
- **Data persistence**: Local storage limitations for large files
- **File handling**: Browser File API integration
- **Routing complexity**: Managing dynamic project IDs

### Phase 2: Enhanced File Management

#### File CRUD Operations
- **File upload with metadata**: Name, type, size, timestamps
- **File editing**: Custom naming and job type modification
- **File deletion**: Confirmation dialogs and cleanup
- **File persistence**: Storage within project context

#### Technical Implementation
```typescript
interface ProjectFile {
    id: string;
    name: string; // Actual filename (no custom names)
    type: string;
    size: number;
    uploadedAt: string;
    status: "uploaded" | "processing" | "completed" | "failed";
    jobType?: JobType;
    jobId?: string;
    downloadUrl?: string;
}
```

#### Key Decisions
- **Simplified naming**: Removed custom file naming to prevent validation issues
- **Direct filename usage**: Always use `file.name` from File API
- **Job type flexibility**: Allow files to be reassigned to different job types

### Phase 3: Job Page Integration

#### Seamless Transitions
- **Project context passing**: Store file data in localStorage for job pages
- **Automatic file loading**: Job pages load files from project context
- **State preservation**: Maintain project association during job processing

#### Context Structure
```typescript
interface ProjectContext {
    projectId: string;
    fileName: string;
    fileType: string;
    fileSize: number;
    fileUrl: string; // Blob URL for file data
    projectType: JobType;
}
```

#### Integration Challenges
- **Blob URL lifecycle**: Managing temporary blob URLs across navigation
- **File reconstruction**: Converting blobs back to File objects on job pages
- **Context cleanup**: Preventing stale data accumulation

### Phase 4: Advanced UI/UX Features

#### Modal Systems
- **Upload Modal**: Job type selection and file upload
- **Edit Modal**: File metadata modification
- **Delete Confirmation**: Safe file removal with user confirmation

#### State Management
- **React hooks**: useState for local state, useEffect for lifecycle
- **Context preservation**: Maintaining state across page transitions
- **Error handling**: Toast notifications and user feedback

## Technical Challenges & Solutions

### Challenge 1: File Validation False Positives

#### Problem
- Custom filenames without extensions caused validation failures
- Backend expected proper file extensions for format detection

#### Solution
- Removed custom file naming system entirely
- Always use actual filename from File API
- Let backend handle file type validation

#### Impact
- Eliminated validation errors from display names
- Simplified user experience
- More reliable file processing

### Challenge 2: Blob URL Persistence

#### Problem
- Blob URLs become invalid after page navigation
- File data lost when continuing translation from project page

#### Solution
- Preserve existing blob URLs in localStorage context
- Check for valid fileUrl before overwriting context
- Maintain file data across navigation

#### Technical Implementation
```typescript
// Preserve existing fileUrl if available
if (context.fileUrl) {
    context.projectId = project.id;
    context.displayName = file.name;
    localStorage.setItem('octavia_project_context', JSON.stringify(context));
    redirectToJobType(project.type);
    return;
}
```

### Challenge 3: Job Type Routing

#### Problem
- Frontend file type validation prevented uploading certain file types for specific jobs
- Users couldn't upload video files for audio translation (even if backend supported it)

#### Solution
- Removed frontend file type restrictions
- Allow all file types for all job types
- Delegate validation to backend APIs

#### Impact
- More flexible file handling
- Better user experience
- Proper error messages from backend

### Challenge 4: State Synchronization

#### Problem
- Multiple browser tabs/windows could have conflicting state
- File operations not reflected across tabs

#### Solution
- Storage event listeners for cross-tab synchronization
- Automatic file list reloading on visibility changes
- Consistent state management

### Challenge 5: File Context Management

#### Problem
- Complex context passing between project and job pages
- Different data structures for different operations

#### Solution
- Unified context structure for all job types
- Clear separation between project files and job processing
- Robust error handling for context parsing

## Integration with Job Pages

### Video Translation Page (`/dashboard/video`)
- **Context Check**: `context.projectType === 'Video Translation'`
- **File Loading**: Creates File object from blob URL
- **Features**: Video processing, lip-sync, multi-language support

### Audio Translation Page (`/dashboard/audio`)
- **Context Check**: `context.projectType === 'Audio Translation'`
- **Integration**: Audio processing with voice selection
- **Features**: Voice cloning, audio format conversion

### Subtitles Page (`/dashboard/subtitles`)
- **Context Check**: `context.projectType === 'Subtitle Generation'`
- **Functionality**: Automatic subtitle generation from media
- **Features**: Multiple format support, timing adjustment

### Subtitle Translation (`/dashboard/subtitles/translate`)
- **Context Check**: `context.projectType === 'Subtitle Translation'`
- **Purpose**: Translate existing subtitle files
- **Features**: Multi-language subtitle translation

### Subtitle to Audio (`/dashboard/audio/subtitle-to-audio`)
- **Context Check**: `context.projectType === 'Subtitle to Audio'`
- **Functionality**: Convert subtitle files to audio narration
- **Features**: Voice synthesis, timing synchronization

## Data Flow Architecture

### File Upload Flow
1. User selects job type in modal
2. User selects file (any type accepted)
3. File validated by backend (not frontend)
4. File blob created and stored in localStorage
5. Project context created with file metadata
6. User redirected to appropriate job page
7. Job page loads file from context and processes

### Continuation Flow
1. User clicks "Continue Translation" on project file
2. System checks for existing file context
3. If found, preserves blob URL and updates project info
4. If not found, user must re-upload file
5. Redirects to job page with preserved file data

### State Management Flow
1. Projects stored in `octavia_projects` localStorage
2. Project files stored in `octavia_project_files_${projectId}`
3. Job context stored in `octavia_project_context`
4. Cross-tab synchronization via storage events

## Performance Considerations

### Optimizations Implemented
- **Lazy loading**: Files loaded only when needed
- **Blob management**: Efficient memory usage with URL cleanup
- **Context pruning**: Automatic cleanup of stale data
- **Incremental updates**: Targeted state updates instead of full reloads

### Memory Management
- **Blob URL lifecycle**: Proper cleanup to prevent memory leaks
- **Context expiration**: Automatic removal of unused contexts
- **File size limits**: Backend-enforced limits prevent memory issues

## Error Handling & User Experience

### Error Scenarios Covered
- **Invalid file types**: Backend validation with clear messages
- **Network failures**: Graceful degradation and retry options
- **Corrupted data**: Fallback to re-upload prompts
- **Permission issues**: Clear user feedback and guidance

### User Feedback Systems
- **Toast notifications**: Success/error messages
- **Loading states**: Progress indicators during operations
- **Confirmation dialogs**: Safe destructive operations
- **Contextual help**: File type guidance and requirements

### Phase 5: Backend Integration and Project-Level Persistence

#### Project ID Integration
- **API Tagging**: All translation endpoints (`/video`, `/audio`, `/subtitles`, etc.) updated to accept an optional `project_id`.
- **Job Persistence**: The `project_id` is stored in the job metadata within the unified job storage (Supabase/Local).
- **Association Flow**: Jobs started from a project context are now permanently linked to that project on the server.

#### Real-time Project Dashboard
- **Backend Job Fetching**: The project detail page now queries the backend (`/api/translate/jobs/history`) to find all jobs associated with the current project ID.
- **Dynamic Job List**: A new "Translation Jobs" section displays active and completed jobs, separate from the original "Project Files".
- **Live Progress Updates**: Uses the backend as the single source of truth for job status and progress.

#### Enhanced User Controls
- **Manual Refresh**: Added a refresh button to trigger an immediate sync between the frontend and the backend status.
- **Direct Downloads**: Implemented a download system within the project view that fetches completed translation results directly from the job ID.

#### Technical Highlights
- **Demo Mode Parity**: Demo users now have access to real job history and file storage within projects, moving away from hardcoded mock data.
- **Event-Driven UI**: LocalStorage updates are synchronized with backend data to ensure the UI is always up-to-date across sessions.

### Challenge 6: Orphaned Jobs

#### Problem
- Jobs started within a project were not visible if the browser session was cleared or if the user logged in from a different device.

#### Solution
- Implemented `project_id` persistence in the backend job storage.
- Updated `loadProjectFiles` to perform a reconciliation between `localStorage` (for speed) and the backend API (for accuracy).

#### Impact
- Truly persistent project history.
- Users can see their "Created Files" even after clearing cache or switching computers.

### Challenge 7: Demo User Data Silos

#### Problem
- Demo users had pre-populated fake data that didn't reflect their actual actions, leading to confusion when they uploaded real files.

#### Solution
- Enabled real project file storage for demo accounts.
- Replaced hardcoded demo history with the actual `/api/translate/jobs/history` endpoint.

## Integration with Job Pages

### Unified Job Tagging
The `ApiService` in `lib/api.ts` now includes `projectId` as a core parameter for:
- `translateVideo(file, targetLanguage, separate, projectId?)`
- `generateSubtitles(file, format, language, projectId?)`
- `translateAudio({ file, sourceLanguage, targetLanguage, projectId? })`
- `translateSubtitleFile({ file, sourceLanguage, targetLanguage, projectId? })`

## Data Flow Architecture (Updated)

### Job Lifecycle
1. **Creation**: Job started with `projectId`.
2. **Backend Storage**: Job entry created in database with `project_id` field.
3. **Frontend Recovery**: Project page filters global job history by `projectId`.
4. **Completion**: Job status moves to `completed`, triggering `download_url` visibility on the project page.

## Conclusion

The Octavia Projects Page has reached full maturity as a professional-grade project management system. By transitioning from a local-only demo to a backend-integrated persistent system, we have ensured that all translation work is traceable, organized, and available across devices.

Key achievements in the latest update:
- **Full Backend-to-Frontend Project ID wiring**.
- **Real job tracking for all users**, including Demo accounts.
- **Centralized "Created Files" management** within the project view.
- **Simplified download and refresh workflows** for end-users.
```