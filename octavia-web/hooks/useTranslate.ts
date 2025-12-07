import { useState, useCallback } from 'react';
import { api, TranslationRequest, TranslationResponse } from '@/lib/api';
import { TranslationProgress } from '@/types/translate';

export const useTranslate = () => {
    const [progress, setProgress] = useState<TranslationProgress>({
        status: 'idle',
        progress: 0,
    });

    const translateVideo = useCallback(async (file: File, targetLanguage: string): Promise<TranslationResponse> => {
        try {
            setProgress({ status: 'uploading', progress: 10, message: 'Starting video translation...' });

            // Call the API to start VIDEO translation (not subtitle)
            const startResult = await api.translateVideo(file, targetLanguage);

            if (!startResult.success || !startResult.job_id) {
                setProgress({
                    status: 'error',
                    progress: 0,
                    message: startResult.error || 'Failed to start video translation'
                });

                setTimeout(() => {
                    setProgress({ status: 'idle', progress: 0 });
                }, 3000);

                return {
                    success: false,
                    error: startResult.error || 'Failed to start video translation',
                };
            }

            const jobId = startResult.job_id;
            setProgress({ status: 'translating', progress: 20, message: 'Translating video...' });

            // Poll for completion
            let attempts = 0;
            const maxAttempts = 300; // 5 minutes for video translation (slower)

            while (attempts < maxAttempts) {
                try {
                    const statusResult = await api.getJobStatus(jobId);

                    if (statusResult.success && statusResult.data) {
                        const { status, progress: jobProgress, download_url, error } = statusResult.data;

                        if (status === 'completed' && download_url) {
                            setProgress({ status: 'downloading', progress: 90, message: 'Downloading translated video...' });

                            // Download the video file
                            try {
                                const fileBlob = await api.downloadFile(jobId);
                                const fileName = `translated_${file.name.replace(/\.(mp4|avi|mov|mkv|webm)$/, '')}_${targetLanguage}.mp4`;

                                api.saveFile(fileBlob, fileName);

                                setProgress({ status: 'complete', progress: 100, message: 'Video translation complete!' });

                                setTimeout(() => {
                                    setProgress({ status: 'idle', progress: 0 });
                                }, 3000);

                                return {
                                    success: true,
                                    file: fileBlob,
                                    fileName,
                                };
                            } catch (downloadError) {
                                setProgress({
                                    status: 'error',
                                    progress: 0,
                                    message: 'Video translation completed but download failed'
                                });

                                setTimeout(() => {
                                    setProgress({ status: 'idle', progress: 0 });
                                }, 3000);

                                return {
                                    success: false,
                                    error: 'Video translation completed but download failed',
                                };
                            }
                        } else if (status === 'failed') {
                            setProgress({
                                status: 'error',
                                progress: 0,
                                message: error || 'Video translation failed'
                            });

                            setTimeout(() => {
                                setProgress({ status: 'idle', progress: 0 });
                            }, 3000);

                            return {
                                success: false,
                                error: error || 'Video translation failed',
                            };
                        } else if (status === 'processing') {
                            // Update progress based on job progress
                            setProgress({
                                status: 'translating',
                                progress: 20 + (jobProgress * 0.7), // 20-90% range
                                message: 'Translating video...'
                            });
                        }
                    }

                    // Wait 2 seconds before next check (video is slower)
                    await new Promise(resolve => setTimeout(resolve, 2000));
                    attempts++;

                } catch (pollError) {
                    console.error('Error polling video translation status:', pollError);
                    attempts++;
                }
            }

            // Timeout
            setProgress({
                status: 'error',
                progress: 0,
                message: 'Video translation timed out'
            });

            setTimeout(() => {
                setProgress({ status: 'idle', progress: 0 });
            }, 3000);

            return {
                success: false,
                error: 'Video translation timed out',
            };

        } catch (error) {
            setProgress({
                status: 'error',
                progress: 0,
                message: error instanceof Error ? error.message : 'Video translation failed unexpectedly'
            });

            setTimeout(() => {
                setProgress({ status: 'idle', progress: 0 });
            }, 3000);

            return {
                success: false,
                error: error instanceof Error ? error.message : 'Unknown error occurred during video translation',
            };
        }
    }, []);

    const translateSubtitle = useCallback(async (data: TranslationRequest): Promise<TranslationResponse> => {
        try {
            setProgress({ status: 'uploading', progress: 10, message: 'Starting subtitle translation...' });

            // Call the API to start subtitle translation (async)
            const startResult = await api.translateSubtitleFile(data);

            if (!startResult.success || !startResult.job_id) {
                setProgress({
                    status: 'error',
                    progress: 0,
                    message: startResult.error || 'Failed to start subtitle translation'
                });

                setTimeout(() => {
                    setProgress({ status: 'idle', progress: 0 });
                }, 3000);

                return {
                    success: false,
                    error: startResult.error || 'Failed to start subtitle translation',
                };
            }

            const jobId = startResult.job_id;
            setProgress({ status: 'translating', progress: 20, message: 'Translating subtitles...' });

            // Poll for completion
            let attempts = 0;
            const maxAttempts = 120; // 2 minutes with 1-second intervals

            while (attempts < maxAttempts) {
                try {
                    const statusResult = await api.getSubtitleTranslationStatus(jobId);

                    if (statusResult.success && statusResult.data) {
                        const { status, progress: jobProgress, download_url, error } = statusResult.data;

                        if (status === 'completed' && download_url) {
                            setProgress({ status: 'downloading', progress: 90, message: 'Downloading translated subtitle...' });

                            // Download the file
                            try {
                                const fileBlob = await api.downloadFileByUrl(`${process.env.NEXT_PUBLIC_API_URL}${download_url}`);
                                const fileName = `translated_${data.file.name.replace(/\.(srt|vtt|ass|ssa)$/, '')}_${data.targetLanguage}.${data.file.name.split('.').pop()}`;

                                api.saveFile(fileBlob, fileName);

                                setProgress({ status: 'complete', progress: 100, message: 'Subtitle translation complete!' });

                                setTimeout(() => {
                                    setProgress({ status: 'idle', progress: 0 });
                                }, 3000);

                                return {
                                    success: true,
                                    file: fileBlob,
                                    fileName,
                                };
                            } catch (downloadError) {
                                setProgress({
                                    status: 'error',
                                    progress: 0,
                                    message: 'Subtitle translation completed but download failed'
                                });

                                setTimeout(() => {
                                    setProgress({ status: 'idle', progress: 0 });
                                }, 3000);

                                return {
                                    success: false,
                                    error: 'Subtitle translation completed but download failed',
                                };
                            }
                        } else if (status === 'failed') {
                            setProgress({
                                status: 'error',
                                progress: 0,
                                message: error || 'Subtitle translation failed'
                            });

                            setTimeout(() => {
                                setProgress({ status: 'idle', progress: 0 });
                            }, 3000);

                            return {
                                success: false,
                                error: error || 'Subtitle translation failed',
                            };
                        } else if (status === 'processing') {
                            // Update progress based on job progress
                            setProgress({
                                status: 'translating',
                                progress: 20 + (jobProgress * 0.7), // 20-90% range
                                message: 'Translating subtitles...'
                            });
                        }
                    }

                    // Wait 1 second before next check
                    await new Promise(resolve => setTimeout(resolve, 1000));
                    attempts++;

                } catch (pollError) {
                    console.error('Error polling subtitle translation status:', pollError);
                    attempts++;
                }
            }

            // Timeout
            setProgress({
                status: 'error',
                progress: 0,
                message: 'Subtitle translation timed out'
            });

            setTimeout(() => {
                setProgress({ status: 'idle', progress: 0 });
            }, 3000);

            return {
                success: false,
                error: 'Subtitle translation timed out',
            };

        } catch (error) {
            setProgress({
                status: 'error',
                progress: 0,
                message: error instanceof Error ? error.message : 'Subtitle translation failed unexpectedly'
            });

            setTimeout(() => {
                setProgress({ status: 'idle', progress: 0 });
            }, 3000);

            return {
                success: false,
                error: error instanceof Error ? error.message : 'Unknown error occurred during subtitle translation',
            };
        }
    }, []);

    // Keep the original translate function for backward compatibility
    const translate = useCallback(async (data: TranslationRequest): Promise<TranslationResponse> => {
        // Default to subtitle translation for backward compatibility
        return translateSubtitle(data);
    }, [translateSubtitle]);

    const resetProgress = useCallback(() => {
        setProgress({ status: 'idle', progress: 0 });
    }, []);

    return {
        translate, // For subtitle translation (backward compatible)
        translateVideo, // For video translation
        translateSubtitle, // For subtitle translation
        progress,
        resetProgress,
    };
};