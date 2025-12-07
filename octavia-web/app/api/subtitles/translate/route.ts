import { NextRequest, NextResponse } from 'next/server';

export const dynamic = 'force-dynamic';

export async function POST(request: NextRequest) {
    try {
        const formData = await request.formData();
        const file = formData.get('subtitleFile') as File;
        const sourceLang = formData.get('sourceLang') as string;
        const targetLang = formData.get('targetLang') as string;

        if (!file) {
            return NextResponse.json(
                { success: false, error: 'No subtitle file provided' },
                { status: 400 }
            );
        }

        if (!sourceLang || !targetLang) {
            return NextResponse.json(
                { success: false, error: 'Source and target languages are required' },
                { status: 400 }
            );
        }

        // Call your Python backend for translation
        const backendUrl = process.env.BACKEND_URL || 'http://localhost:8000';
        const backendFormData = new FormData();
        backendFormData.append('file', file);
        backendFormData.append('source_language', sourceLang);
        backendFormData.append('target_language', targetLang);

        // Get auth token from request headers
        const authHeader = request.headers.get('Authorization');
        
        const response = await fetch(`${backendUrl}/api/translate/subtitle-file`, {
            method: 'POST',
            body: backendFormData,
            headers: authHeader ? {
                'Authorization': authHeader
            } : {},
        });

        if (!response.ok) {
            const errorText = await response.text();
            console.error('Backend translation failed:', errorText);
            return NextResponse.json(
                { success: false, error: 'Translation failed on backend' },
                { status: 500 }
            );
        }

        const result = await response.json();
        
        if (!result.success) {
            return NextResponse.json(
                { success: false, error: result.error || 'Translation failed' },
                { status: 500 }
            );
        }

        // If we have a download_url, download the file
        if (result.download_url) {
            const downloadResponse = await fetch(`${backendUrl}${result.download_url}`, {
                headers: authHeader ? {
                    'Authorization': authHeader
                } : {},
            });

            if (!downloadResponse.ok) {
                throw new Error('Failed to download translated file');
            }

            const translatedBlob = await downloadResponse.blob();
            const translatedFileName = downloadResponse.headers
                .get('Content-Disposition')
                ?.split('filename=')[1]
                ?.replace(/"/g, '') || `translated_${file.name}`;

            return new NextResponse(translatedBlob, {
                headers: {
                    'Content-Type': 'application/octet-stream',
                    'Content-Disposition': `attachment; filename="${translatedFileName}"`,
                },
            });
        }

        // If we have a job_id, poll for completion
        if (result.job_id) {
            let attempts = 0;
            const maxAttempts = 60; // 60 seconds timeout
            
            while (attempts < maxAttempts) {
                await new Promise(resolve => setTimeout(resolve, 1000));
                
                const statusResponse = await fetch(
                    `${backendUrl}/api/translate/subtitle-status/${result.job_id}`,
                    {
                        headers: authHeader ? {
                            'Authorization': authHeader
                        } : {},
                    }
                );
                
                if (statusResponse.ok) {
                    const statusData = await statusResponse.json();
                    
                    if (statusData.success) {
                        if (statusData.status === 'completed' && statusData.download_url) {
                            const downloadResponse = await fetch(
                                `${backendUrl}${statusData.download_url}`,
                                {
                                    headers: authHeader ? {
                                        'Authorization': authHeader
                                    } : {},
                                }
                            );

                            if (!downloadResponse.ok) {
                                throw new Error('Failed to download translated file');
                            }

                            const translatedBlob = await downloadResponse.blob();
                            const translatedFileName = downloadResponse.headers
                                .get('Content-Disposition')
                                ?.split('filename=')[1]
                                ?.replace(/"/g, '') || `translated_${file.name}`;

                            return new NextResponse(translatedBlob, {
                                headers: {
                                    'Content-Type': 'application/octet-stream',
                                    'Content-Disposition': `attachment; filename="${translatedFileName}"`,
                                },
                            });
                        } else if (statusData.status === 'failed') {
                            return NextResponse.json(
                                { success: false, error: statusData.error || 'Translation job failed' },
                                { status: 500 }
                            );
                        }
                    }
                }
                
                attempts++;
            }
            
            return NextResponse.json(
                { success: false, error: 'Translation timed out' },
                { status: 500 }
            );
        }

        return NextResponse.json(
            { success: false, error: 'No download URL or job ID received' },
            { status: 500 }
        );

    } catch (error) {
        console.error('Translation error:', error);
        return NextResponse.json(
            { 
                success: false, 
                error: 'Translation failed', 
                details: error instanceof Error ? error.message : 'Unknown error' 
            },
            { status: 500 }
        );
    }
}