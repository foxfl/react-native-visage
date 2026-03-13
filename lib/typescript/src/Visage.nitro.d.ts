import type { HybridObject } from 'react-native-nitro-modules';
export interface FaceRect {
    x: number;
    y: number;
    width: number;
    height: number;
}
export interface FaceDetection {
    embedding: ArrayBuffer;
    faceRect: FaceRect;
    confidence: number;
}
export interface MatchResult {
    embeddingIndex: number;
    faceRect: FaceRect;
    similarity: number;
}
export interface Album {
    id: string;
    name: string;
    count: number;
}
export type MatchMode = 'any' | 'all' | 'exact';
export type ModelNormalization = 'neg_one_one' | 'zero_one';
export interface ModelConfig {
    /** Absolute path or file:// URI to a compiled .mlmodelc directory */
    modelUri: string;
    /** Output embedding dimension, e.g. 512 */
    embeddingSize: number;
    /** Output feature name. Omit to auto-detect (first model output). */
    outputLayerName?: string;
    /** Square input side length in pixels. Default: 112 */
    inputSize?: number;
    /** Pixel normalization applied before inference. Default: 'neg_one_one' */
    normalization?: ModelNormalization;
}
export interface ScanOptions {
    embeddings: ArrayBuffer[];
    since?: number;
    threshold?: number;
    albumId?: string;
    matchMode?: MatchMode;
    onProgress: (scanned: number, total: number) => void;
    onMatch: (assetId: string, matches: MatchResult[]) => void;
    onComplete: () => void;
    onError: (message: string) => void;
}
export interface Visage extends HybridObject<{
    ios: 'swift';
    android: 'kotlin';
}> {
    extractEmbeddings(imageUri: string): Promise<FaceDetection[]>;
    compareFaces(a: ArrayBuffer, b: ArrayBuffer): Promise<number>;
    scanLibrary(options: ScanOptions): Promise<void>;
    cancelScan(): Promise<void>;
    getAlbums(): Promise<Album[]>;
    setModel(config: ModelConfig): Promise<void>;
}
//# sourceMappingURL=Visage.nitro.d.ts.map