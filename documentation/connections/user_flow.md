# Octavia Cloud Platform - User Flow & Connections

This document outlines the user flow and connections between pages in the Octavia SaaS platform. It visualizes how users navigate through the application and describes the purpose of each page.

## 1. High-Level User Flow (Mermaid Diagram)

```mermaid
graph TD
    %% Public Area
    Landing[Landing Page] -->|Login/Signup| Auth["Auth (Clerk)"]
    Auth -->|Success| Hub["Hub / Dashboard"]

    %% Core Hub Navigation
    Hub -->|Select Video Tool| VideoInput[Video Translator Input]
    Hub -->|Select Audio Tool| AudioInput[Audio Translator]
    Hub -->|Select Subtitle Tool| SubtitleInput[Subtitle Gen Input]
    Hub -->|Manage Account| Settings[General Settings]
    Hub -->|View History| JobHistory[Job History]
    Hub -->|Manage Billing| Billing[Plans & Billing]

    %% Video Translation Flow
    VideoInput -->|Upload & Start| VideoProgress[Translation Progress]
    VideoProgress -->|Complete| VideoReview[Review & Export]
    VideoReview -->|Download| End([User Downloads File])
    VideoReview -->|Back to Hub| Hub

    %% Audio Translation Flow
    AudioInput -->|Upload & Start| AudioProgress[Audio Progress]
    AudioProgress -->|Complete| AudioReview[Audio Review]
    AudioReview -->|Download| End

    %% Subtitle Generation Flow
    SubtitleInput -->|Upload & Start| SubtitleProgress[Subtitle Gen Progress]
    SubtitleProgress -->|Complete| SubtitleReview[Subtitle Review]
    SubtitleReview -->|Export SRT| End
    SubtitleReview -->|Translate SRT| SubtitleTranslator[Subtitle Translator]

    %% Settings & Account Flow
    Settings -->|Advanced| AdvancedSettings[Advanced Settings]
    Settings -->|Profile| Profile[Profile & Security]
    Settings -->|Team| Team[Team / Organization]
    Settings -->|My Voices| MyVoices[My Voices]

    %% Magic Mode Connections
    VideoInput -.->|Enable Magic Mode| MyVoices
    MyVoices -.->|Use Cloned Voice| VideoInput
```

## 2. Page Descriptions & Connections

### Public & Auth
- **Landing Page**: The entry point for new users. Showcases features, pricing, and "Get Started" buttons. Connects to -> **Auth**.
- **Auth (Clerk)**: Handles Login, Sign Up, and Forgot Password. On success, redirects to -> **Hub**.

### Core Dashboard
- **Hub / Dashboard**: The central command center.
  - **Purpose**: Gives quick access to all tools, shows recent activity, and credit balance.
  - **Connections**: Links to all major tools (Video, Audio, Subtitles), Settings, and Billing.

### Video Translation Module
- **Video Translator Input**:
  - **Purpose**: User uploads video, selects languages, and toggles "Magic Mode" (Cloning/Ducking).
  - **Connections**: -> **Translation Progress** (on start).
- **Translation Progress**:
  - **Purpose**: Real-time visualization of the pipeline (e.g., "Cloning Voice...", "Dubbing..."). Shows "Sample Chunk" preview.
  - **Connections**: -> **Review & Export** (on completion).
- **Review & Export**:
  - **Purpose**: Play the final video, view sync stats, and download files.
  - **Connections**: -> **Hub** (to start over).

### Audio Module
- **Audio Translator**:
  - **Purpose**: Similar to Video Input but for audio-only files (podcasts, lectures).
  - **Connections**: -> **Audio Progress**.
- **Subtitle to Audio**:
  - **Purpose**: Converts an uploaded SRT file into spoken audio (TTS).
  - **Connections**: -> **Audio Review**.

### Subtitle Module
- **Subtitle Gen Input**:
  - **Purpose**: Generate subtitles from a video/audio file without translation (transcription).
  - **Connections**: -> **Subtitle Gen Progress**.
- **Subtitle Review**:
  - **Purpose**: Editor interface to correct text and timing before export.
  - **Connections**: -> **Subtitle Translator** (if user wants to translate the generated subs).
- **Subtitle Translator**:
  - **Purpose**: Translate an existing SRT file to another language.

### Settings & Account
- **General Settings**: App preferences (theme, notification settings).
- **Advanced Settings**: Technical toggles for the AI pipeline (e.g., "Always use Magic Mode").
- **My Voices**:
  - **Purpose**: Library of cloned voices. Users can record samples or upload files to create new voice clones here.
  - **Connections**: Referenced by **Video Translator Input** for speaker assignment.
- **Plans & Billing**: Manage subscription, view invoices, and buy credits.
- **Profile & Security**: User details, password, MFA (via Clerk).
- **Team / Organization**: Invite team members and manage roles (via Clerk).
