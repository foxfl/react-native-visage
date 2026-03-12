# react-native-visage

## What this is

A **stateless, on-device face embedding library** for React Native, built with Nitro Modules. It detects faces in images and returns embedding vectors. That's it. No storage, no identity management, no UI. Pure computation primitive.

The consumer is responsible for storing embeddings, managing identity, and deciding what to do with matches.

## Architecture

Built with `react-native-nitro-modules` for direct Swift ↔ C++ interop on iOS and Kotlin on Android. No Objective-C anywhere.

**iOS:** Vision framework for face detection → CoreML-based embedding model → cosine similarity  
**Android:** ML Kit face detection → bundled TFLite MobileFaceNet model → cosine similarity

## Repo structure

```
react-native-visage/
├── nitrogen/          # Nitro-generated bindings — DO NOT edit manually
├── src/               # TypeScript interface and types
├── ios/               # Swift implementation
│   └── Visage.swift
├── android/           # Kotlin implementation
│   └── Visage.kt
├── example/           # Test app — used for manual and automated testing
├── nitro.json         # Nitro module spec — source of truth for the API
├── package.json
└── README.md
```

## Core API

```typescript
// Extract all face embeddings from an image
// Returns one FaceDetection per face found — could be 0, 1, or many
FaceLibrary.extractEmbeddings(imageUri: string): Promise<FaceDetection[]>

// Cosine similarity between two embeddings — utility for consumer-side logic
FaceLibrary.compareFaces(a: Embedding, b: Embedding): Promise<number>

// Scan the device photo library, streaming matches
FaceLibrary.scanLibrary(options: ScanOptions): Promise<void>

// Cancel an in-progress scan
FaceLibrary.cancelScan(): Promise<void>
```

## Core types

```typescript
type Embedding = Float32Array  // opaque — consumer stores and passes back

interface FaceDetection {
  embedding: Embedding
  faceRect: { x: number; y: number; width: number; height: number }
  confidence: number          // 0-1, detection confidence
}

interface MatchResult {
  embeddingIndex: number      // index into the embeddings[] array passed to scanLibrary
  faceRect: { x: number; y: number; width: number; height: number }
  similarity: number          // 0-1, cosine similarity
}

interface ScanOptions {
  embeddings: Embedding[]     // all embeddings to match against — supports multiple people
  since?: number              // unix timestamp — only scan photos newer than this
  threshold?: number          // similarity cutoff, default 0.75
  onProgress: (scanned: number, total: number) => void
  onMatch: (assetId: string, matches: MatchResult[]) => void
  onComplete: () => void
  onError: (error: Error) => void
}
```

## Key design decisions

**Stateless by design.** The library has no internal storage. Embeddings are returned as `Float32Array` blobs and the consumer stores them however they want (MMKV, SQLite, AsyncStorage). This keeps the library a pure primitive with no opinions about identity or persistence.

**embeddingIndex for identity mapping.** `MatchResult.embeddingIndex` maps back to the index in the `embeddings[]` array passed to `scanLibrary`. The consumer knows which embedding belongs to which person — the library doesn't need to.

**Multiple faces per photo are fully supported.** `extractEmbeddings` returns all detected faces. `scanLibrary` matches all passed embeddings per photo, so couple photos, group shots, and family photos all work naturally. `onMatch` fires once per photo with a `MatchResult[]` containing one entry per person found.

**Float32Array over base64.** Embeddings must be passed and returned as `Float32Array` via Nitro's native ArrayBuffer support. Never serialize to base64 — this would destroy performance during library scans.

**`since` timestamp is critical for production use.** Always persist the timestamp of the last successful scan and pass it back in. Scanning the full library on every background run is not acceptable.

## What this library is NOT responsible for

- Storing or persisting embeddings
- Mapping embeddings to named people
- UI of any kind
- Shared albums or photo sharing
- Face clustering of unknown faces
- Real-time camera feed recognition (use react-native-vision-camera for that)
- Liveness detection

## Platform notes

**iOS minimum: iOS 18** — required for the Float16 mlprogram CoreML model format.  
**Android minimum: API 24 (Android 7.0)** — required for ML Kit and TFLite.

The TFLite MobileFaceNet model is bundled inside the Android package. Do not replace it without re-validating embedding compatibility — stored embeddings from the old model will not be comparable to embeddings from a new model.

## Nitro spec is the source of truth

`nitro.json` defines the API. If you change the API:
1. Update `nitro.json` first
2. Run `npx nitrogen` to regenerate bindings in `nitrogen/`
3. Update Swift and Kotlin implementations to match
4. Update TypeScript types in `src/`
5. Never edit files in `nitrogen/` manually — they will be overwritten

## Testing

The `example/` app is the primary testing surface. It covers:
- Single face enrollment from a selfie
- Multi-face extraction from a group photo
- Library scan with progress and match streaming
- Cancellation mid-scan
- Edge cases: no face detected, multiple faces, low confidence

Run the example app on a real device. Simulator performance for ML workloads is not representative.

## Cosine similarity thresholds

These are approximate starting points validated against the bundled models. Tune per use case:

| Use case | Suggested threshold |
|---|---|
| High precision (fewer false positives) | 0.82 |
| Default / balanced | 0.75 |
| High recall (catch more, accept more false positives) | 0.65 |

## Open-source considerations

This library is intentionally scoped to be a general-purpose primitive. The face embedding + stateless design means it is useful for any app that needs face matching — not just the couples photo use case that motivated it. Keep PRs and issues scoped to the core primitive. Feature requests for identity management, UI, or storage belong in consuming apps, not here.