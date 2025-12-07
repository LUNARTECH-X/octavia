export interface LanguageOption {
    value: string;
    label: string;
    code: string;
}

export const LANGUAGES: LanguageOption[] = [
    { value: 'english', label: 'English', code: 'en' },
    { value: 'spanish', label: 'Spanish', code: 'es' },
    { value: 'french', label: 'French', code: 'fr' },
    { value: 'german', label: 'German', code: 'de' },
    { value: 'italian', label: 'Italian', code: 'it' },
    { value: 'portuguese', label: 'Portuguese', code: 'pt' },
    { value: 'russian', label: 'Russian', code: 'ru' },
    { value: 'japanese', label: 'Japanese', code: 'ja' },
    { value: 'korean', label: 'Korean', code: 'ko' },
    { value: 'chinese', label: 'Chinese', code: 'zh-cn' },
    { value: 'arabic', label: 'Arabic', code: 'ar' },
    { value: 'hindi', label: 'Hindi', code: 'hi' },
];

export interface TranslationProgress {
    status: 'idle' | 'uploading' | 'translating' | 'downloading' | 'complete' | 'error';
    progress: number;
    message?: string;
}