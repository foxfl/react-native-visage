"use strict";

import { NitroModules } from 'react-native-nitro-modules';
const VisageModule = NitroModules.createHybridObject('Visage');
export const FaceLibrary = {
  async extractEmbeddings(imageUri) {
    const detections = await VisageModule.extractEmbeddings(imageUri);
    return detections.map(d => ({
      ...d,
      embedding: new Float32Array(d.embedding)
    }));
  },
  async compareFaces(a, b) {
    return VisageModule.compareFaces(a.buffer, b.buffer);
  },
  async scanLibrary(options) {
    return VisageModule.scanLibrary({
      embeddings: options.embeddings.map(e => e.buffer),
      since: options.since,
      threshold: options.threshold,
      albumId: options.albumId,
      matchMode: options.matchMode,
      onProgress: options.onProgress,
      onMatch: options.onMatch,
      onComplete: options.onComplete,
      onError: message => options.onError(new Error(message))
    });
  },
  async cancelScan() {
    return VisageModule.cancelScan();
  },
  async getAlbums() {
    return VisageModule.getAlbums();
  },
  async setModel(config) {
    return VisageModule.setModel(config);
  }
};
//# sourceMappingURL=index.js.map