import { NitroModules } from 'react-native-nitro-modules';
import type { Visage } from './Visage.nitro';

export type {
  FaceRect,
  FaceDetection,
  MatchResult,
  ScanOptions,
  Album,
  MatchMode,
  ModelConfig,
  ModelNormalization,
} from './Visage.nitro';

const VisageModule = NitroModules.createHybridObject<Visage>('Visage');

export type Embedding = Float32Array;

export interface ScanLibraryOptions {
  embeddings: Embedding[];
  since?: number;
  threshold?: number;
  albumId?: string;
  matchMode?: import('./Visage.nitro').MatchMode;
  onProgress: (scanned: number, total: number) => void;
  onMatch: (
    assetId: string,
    matches: import('./Visage.nitro').MatchResult[]
  ) => void;
  onComplete: () => void;
  onError: (error: Error) => void;
}

export const FaceLibrary = {
  async extractEmbeddings(imageUri: string) {
    const detections = await VisageModule.extractEmbeddings(imageUri);
    return detections.map((d) => ({
      ...d,
      embedding: new Float32Array(d.embedding) as Embedding,
    }));
  },

  async compareFaces(a: Embedding, b: Embedding): Promise<number> {
    return VisageModule.compareFaces(
      a.buffer as ArrayBuffer,
      b.buffer as ArrayBuffer
    );
  },

  async scanLibrary(options: ScanLibraryOptions): Promise<void> {
    return VisageModule.scanLibrary({
      embeddings: options.embeddings.map((e) => e.buffer as ArrayBuffer),
      since: options.since,
      threshold: options.threshold,
      albumId: options.albumId,
      matchMode: options.matchMode,
      onProgress: options.onProgress,
      onMatch: options.onMatch,
      onComplete: options.onComplete,
      onError: (message: string) => options.onError(new Error(message)),
    });
  },

  async cancelScan(): Promise<void> {
    return VisageModule.cancelScan();
  },

  async getAlbums() {
    return VisageModule.getAlbums();
  },

  async setModel(config: import('./Visage.nitro').ModelConfig): Promise<void> {
    return VisageModule.setModel(config);
  },
};
