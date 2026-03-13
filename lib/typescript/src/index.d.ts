export type { FaceRect, FaceDetection, MatchResult, ScanOptions, Album, MatchMode, ModelConfig, ModelNormalization, } from './Visage.nitro';
export type Embedding = Float32Array;
export interface ScanLibraryOptions {
    embeddings: Embedding[];
    since?: number;
    threshold?: number;
    albumId?: string;
    matchMode?: import('./Visage.nitro').MatchMode;
    onProgress: (scanned: number, total: number) => void;
    onMatch: (assetId: string, matches: import('./Visage.nitro').MatchResult[]) => void;
    onComplete: () => void;
    onError: (error: Error) => void;
}
export declare const FaceLibrary: {
    extractEmbeddings(imageUri: string): Promise<{
        embedding: Embedding;
        faceRect: import("./Visage.nitro").FaceRect;
        confidence: number;
    }[]>;
    compareFaces(a: Embedding, b: Embedding): Promise<number>;
    scanLibrary(options: ScanLibraryOptions): Promise<void>;
    cancelScan(): Promise<void>;
    getAlbums(): Promise<import("./Visage.nitro").Album[]>;
    setModel(config: import("./Visage.nitro").ModelConfig): Promise<void>;
};
//# sourceMappingURL=index.d.ts.map