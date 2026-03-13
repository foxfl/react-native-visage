import Foundation
import UIKit
import Vision
import Photos
import Accelerate
import NitroModules

class Visage: HybridVisageSpec {

    // MARK: - Properties

    private var isCancelled = false
    private let processingQueue = DispatchQueue(
        label: "com.visage.processing",
        qos: .userInitiated
    )
    private var faceEmbedder = FaceEmbedder()

    // MARK: - extractEmbeddings

    func extractEmbeddings(imageUri: String) throws -> Promise<[FaceDetection]> {
        return Promise.async { [self] in
            let cgImage = try self.loadImage(from: imageUri)
            return try self.detectAndEmbed(in: cgImage)
        }
    }

    // MARK: - compareFaces

    func compareFaces(a: ArrayBuffer, b: ArrayBuffer) throws -> Promise<Double> {
        return Promise.async {
            let embSize = self.faceEmbedder.embeddingSize
            let sizeA = a.size / MemoryLayout<Float>.size
            let sizeB = b.size / MemoryLayout<Float>.size
            guard sizeA == embSize, sizeB == embSize else {
                throw VisageError.embeddingDimensionMismatch(
                    expected: embSize,
                    got: Int(sizeA != embSize ? sizeA : sizeB)
                )
            }

            let ptrA = UnsafeMutableRawPointer(a.data).bindMemory(to: Float.self, capacity: Int(sizeA))
            let ptrB = UnsafeMutableRawPointer(b.data).bindMemory(to: Float.self, capacity: Int(sizeB))

            return self.cosineSimilarity(ptrA, ptrB, count: embSize)
        }
    }

    // MARK: - scanLibrary

    func scanLibrary(options: ScanOptions) throws -> Promise<Void> {
        return Promise.async { [self] in
            self.isCancelled = false

            // Request photo library authorization
            let status = await self.requestPhotoAccess()
            guard status == .authorized || status == .limited else {
                options.onError("Photo library access denied")
                return
            }

            // Build fetch options
            let fetchOptions = PHFetchOptions()
            fetchOptions.sortDescriptors = [NSSortDescriptor(key: "creationDate", ascending: true)]

            if let since = options.since {
                let sinceDate = Date(timeIntervalSince1970: since)
                fetchOptions.predicate = NSPredicate(format: "creationDate > %@", sinceDate as NSDate)
            }

            let assets: PHFetchResult<PHAsset>
            if let albumId = options.albumId,
               let collection = PHAssetCollection.fetchAssetCollections(
                   withLocalIdentifiers: [albumId], options: nil
               ).firstObject {
                assets = PHAsset.fetchAssets(in: collection, options: fetchOptions)
            } else {
                assets = PHAsset.fetchAssets(with: .image, options: fetchOptions)
            }
            let total = Double(assets.count)
            options.onProgress(0, total)

            if assets.count == 0 {
                options.onComplete()
                return
            }

            // Copy reference embeddings from ArrayBuffers into Float arrays for comparison
            let threshold = options.threshold ?? 0.75
            let referenceEmbeddings: [[Float]] = options.embeddings.map { buffer in
                let count = buffer.size / MemoryLayout<Float>.size
                let ptr = UnsafeMutableRawPointer(buffer.data).bindMemory(to: Float.self, capacity: Int(count))
                return Array(UnsafeBufferPointer(start: ptr, count: Int(count)))
            }

            // Process assets on background queue
            await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
                self.processingQueue.async {
                    let imageManager = PHImageManager.default()
                    let imageOptions = PHImageRequestOptions()
                    imageOptions.isSynchronous = true
                    imageOptions.deliveryMode = .highQualityFormat
                    imageOptions.resizeMode = .exact
                    // Do not allow network access — photos requiring iCloud download cause a
                    // PHImageManager result-accumulator timeout (3s) followed by a crash when
                    // the delayed delivery callback fires into a torn-down synchronous request.
                    // iCloud-only photos are skipped (image == nil → guard fails → continue)
                    // and will be picked up on the next scan once they are downloaded locally.
                    imageOptions.isNetworkAccessAllowed = false

                    for i in 0..<assets.count {
                        if self.isCancelled {
                            continuation.resume()
                            return
                        }

                        autoreleasepool {
                            let asset = assets.object(at: i)

                            // Use the longer dimension of the asset to preserve aspect ratio
                            // and ensure faces are large enough for reliable detection.
                            // Cap at 1024 to keep memory and CPU reasonable.
                            let longerSide = min(max(asset.pixelWidth, asset.pixelHeight), 1024)
                            let targetSize = CGSize(width: longerSide, height: longerSide)

                            imageManager.requestImage(
                                for: asset,
                                targetSize: targetSize,
                                contentMode: .aspectFit,
                                options: imageOptions
                            ) { image, _ in
                                guard let uiImage = image,
                                      let cgImage = try? self.normalizeOrientation(uiImage) else { return }

                                do {
                                    let detections = try self.detectAndEmbed(in: cgImage)
                                    var matchResults: [MatchResult] = []

                                    for detection in detections {
                                                        let embSize = self.faceEmbedder.embeddingSize
                                        let embPtr = UnsafeMutableRawPointer(detection.embedding.data).bindMemory(
                                            to: Float.self,
                                            capacity: embSize
                                        )

                                        for (refIdx, refEmb) in referenceEmbeddings.enumerated() {
                                            refEmb.withUnsafeBufferPointer { refBuf in
                                                guard let refPtr = refBuf.baseAddress else { return }
                                                let similarity = self.cosineSimilarity(
                                                    embPtr, refPtr, count: embSize
                                                )
                                                NSLog("[Visage] asset=%@ ref=%d sim=%.4f threshold=%.2f",
                                                      asset.localIdentifier, refIdx, similarity, threshold)
                                                if similarity >= threshold {
                                                    matchResults.append(MatchResult(
                                                        embeddingIndex: Double(refIdx),
                                                        faceRect: detection.faceRect,
                                                        similarity: similarity
                                                    ))
                                                }
                                            }
                                        }
                                    }

                                    if detections.isEmpty {
                                        NSLog("[Visage] asset=%@ — no faces detected", asset.localIdentifier)
                                    }

                                    let mode = options.matchMode ?? .any
                                    NSLog("[Visage] asset=%@ mode=%@ detections=%d matchResults=%d enrolled=%d",
                                          asset.localIdentifier, mode.stringValue,
                                          detections.count, matchResults.count, referenceEmbeddings.count)

                                    let shouldReport: Bool
                                    switch mode {
                                    case .any:
                                        // At least one enrolled face found — extra people in photo are fine
                                        shouldReport = !matchResults.isEmpty
                                    case .all:
                                        // Every enrolled face must have at least one match — extra people are fine
                                        let matchedIndices = Set(matchResults.map { Int($0.embeddingIndex) })
                                        shouldReport = matchedIndices.count == referenceEmbeddings.count
                                    case .exact:
                                        // Every enrolled face must match AND no extra unmatched faces in photo
                                        let matchedIndices = Set(matchResults.map { Int($0.embeddingIndex) })
                                        let allEnrolledFound = matchedIndices.count == referenceEmbeddings.count
                                        let noExtraFaces = detections.count == referenceEmbeddings.count
                                        shouldReport = allEnrolledFound && noExtraFaces
                                    @unknown default:
                                        shouldReport = !matchResults.isEmpty
                                    }

                                    if shouldReport {
                                        options.onMatch(asset.localIdentifier, matchResults)
                                    }
                                } catch {
                                    // Per-asset errors are non-fatal — log and continue
                                    NSLog("[Visage] Error processing asset \(asset.localIdentifier): \(error)")
                                }
                            }

                            options.onProgress(Double(i + 1), total)
                        }
                    }

                    if !self.isCancelled {
                        options.onComplete()
                    }
                    continuation.resume()
                }
            }
        }
    }

    // MARK: - cancelScan

    func cancelScan() throws -> Promise<Void> {
        return Promise.async {
            self.isCancelled = true
        }
    }

    // MARK: - getAlbums

    func getAlbums() throws -> Promise<[Album]> {
        return Promise.async {
            var albums: [Album] = []

            // User albums
            let userAlbums = PHAssetCollection.fetchAssetCollections(
                with: .album, subtype: .any, options: nil
            )
            userAlbums.enumerateObjects { collection, _, _ in
                let count = PHAsset.fetchAssets(in: collection, options: nil).count
                if count > 0 {
                    albums.append(Album(
                        id: collection.localIdentifier,
                        name: collection.localizedTitle ?? "Untitled",
                        count: Double(count)
                    ))
                }
            }

            // Smart albums (Camera Roll, Favorites, etc.)
            let smartAlbums = PHAssetCollection.fetchAssetCollections(
                with: .smartAlbum, subtype: .any, options: nil
            )
            smartAlbums.enumerateObjects { collection, _, _ in
                let fetchOptions = PHFetchOptions()
                fetchOptions.predicate = NSPredicate(format: "mediaType == %d", PHAssetMediaType.image.rawValue)
                let count = PHAsset.fetchAssets(in: collection, options: fetchOptions).count
                if count > 0 {
                    albums.append(Album(
                        id: collection.localIdentifier,
                        name: collection.localizedTitle ?? "Untitled",
                        count: Double(count)
                    ))
                }
            }

            return albums.sorted { $0.name < $1.name }
        }
    }

    // MARK: - setModel

    func setModel(config: ModelConfig) throws -> Promise<Void> {
        return Promise.async { [self] in
            let url: URL
            if config.modelUri.hasPrefix("file://") {
                guard let u = URL(string: config.modelUri) else {
                    throw VisageError.invalidImageUri(config.modelUri)
                }
                url = u
            } else {
                url = URL(fileURLWithPath: config.modelUri)
            }

            let norm: FaceEmbedder.NormalizationMode =
                config.normalization == .zeroOne ? .zeroOne : .negOneOne

            self.faceEmbedder = FaceEmbedder(
                modelURL: url,
                embeddingSize: Int(config.embeddingSize),
                inputSize: Int(config.inputSize ?? 112),
                outputLayerName: config.outputLayerName,
                normalization: norm
            )
        }
    }

    // MARK: - Private helpers

    /// Load a CGImage from a URI string with EXIF orientation baked in.
    /// CGImageSourceCreateImageAtIndex ignores EXIF rotation, so face rect
    /// coordinates would be in the raw pixel space rather than the visual space
    /// that React Native displays. Using UIImage + UIGraphicsImageRenderer
    /// redraws the pixels in the correct orientation so coordinates align.
    private func loadImage(from uri: String) throws -> CGImage {
        let uiImage: UIImage

        if uri.hasPrefix("file://"), let path = URL(string: uri)?.path {
            guard let img = UIImage(contentsOfFile: path) else {
                throw VisageError.invalidImageUri(uri)
            }
            uiImage = img
        } else if uri.hasPrefix("/") {
            guard let img = UIImage(contentsOfFile: uri) else {
                throw VisageError.invalidImageUri(uri)
            }
            uiImage = img
        } else if let url = URL(string: uri),
                  let data = try? Data(contentsOf: url),
                  let img = UIImage(data: data) {
            uiImage = img
        } else {
            throw VisageError.invalidImageUri(uri)
        }

        return try normalizeOrientation(uiImage)
    }

    /// Redraw a UIImage into a CGImage with EXIF orientation applied.
    /// This ensures Vision face coordinates are in the visual coordinate space.
    ///
    /// scale = 1.0 is critical: UIGraphicsImageRenderer defaults to the screen
    /// scale (3× on modern iPhones), which would produce a 9× larger image
    /// (9072×12096 instead of 3024×4032) and corrupt the embedding quality.
    private func normalizeOrientation(_ uiImage: UIImage) throws -> CGImage {
        if uiImage.imageOrientation == .up, let cg = uiImage.cgImage { return cg }
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1.0
        let renderer = UIGraphicsImageRenderer(size: uiImage.size, format: format)
        let normalized = renderer.image { _ in uiImage.draw(at: .zero) }
        guard let cg = normalized.cgImage else {
            throw VisageError.invalidImageUri("orientation normalization failed")
        }
        return cg
    }

    /// Run face detection and embedding extraction on a CGImage.
    /// Returns empty array if no faces found — does not throw.
    private func detectAndEmbed(in cgImage: CGImage) throws -> [FaceDetection] {
        let imageWidth = CGFloat(cgImage.width)
        let imageHeight = CGFloat(cgImage.height)

        // Vision face detection
        let request = VNDetectFaceRectanglesRequest()
        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        try handler.perform([request])

        guard let results = request.results, !results.isEmpty else {
            return []
        }

        var detections: [FaceDetection] = []

        for observation in results {
            let bbox = observation.boundingBox

            // Vision uses normalized coordinates with origin at bottom-left.
            // Convert to pixel coordinates with origin at top-left.
            let x = bbox.origin.x * imageWidth
            let y = (1.0 - bbox.origin.y - bbox.height) * imageHeight
            let w = bbox.width * imageWidth
            let h = bbox.height * imageHeight

            // Add 20% padding for hair/chin context, clamped to image bounds
            let padding: CGFloat = 0.2
            let padW = w * padding
            let padH = h * padding
            let cropRect = CGRect(
                x: max(0, x - padW),
                y: max(0, y - padH),
                width: min(imageWidth - max(0, x - padW), w + 2 * padW),
                height: min(imageHeight - max(0, y - padH), h + 2 * padH)
            ).integral

            guard let croppedImage = cgImage.cropping(to: cropRect) else {
                continue
            }

            do {
                let embeddingFloats = try faceEmbedder.getEmbedding(from: croppedImage)

                // Pack Float array into an ArrayBuffer
                let bufferSize = embeddingFloats.count * MemoryLayout<Float>.size
                let buffer = ArrayBuffer.allocate(size: bufferSize)
                let floatPtr = UnsafeMutableRawPointer(buffer.data).bindMemory(to: Float.self, capacity: embeddingFloats.count)
                for i in 0..<embeddingFloats.count {
                    floatPtr[i] = embeddingFloats[i]
                }

                let faceRect = FaceRect(x: Double(x), y: Double(y), width: Double(w), height: Double(h))
                let detection = FaceDetection(
                    embedding: buffer,
                    faceRect: faceRect,
                    confidence: Double(observation.confidence)
                )
                detections.append(detection)
            } catch {
                // Skip faces where embedding extraction fails
                NSLog("[Visage] Embedding extraction failed for a face: \(error)")
                continue
            }
        }

        return detections
    }

    /// Compute cosine similarity between two Float vectors using Accelerate.
    private func cosineSimilarity(
        _ a: UnsafePointer<Float>,
        _ b: UnsafePointer<Float>,
        count: Int
    ) -> Double {
        var dot: Float = 0
        var normA: Float = 0
        var normB: Float = 0

        vDSP_dotpr(a, 1, b, 1, &dot, vDSP_Length(count))
        vDSP_svesq(a, 1, &normA, vDSP_Length(count))
        vDSP_svesq(b, 1, &normB, vDSP_Length(count))

        let denom = sqrt(normA) * sqrt(normB)
        guard denom > 0 else { return 0.0 }
        return Double(dot / denom)
    }

    /// Request photo library access (iOS 14+).
    private func requestPhotoAccess() async -> PHAuthorizationStatus {
        return await withCheckedContinuation { continuation in
            PHPhotoLibrary.requestAuthorization(for: .readWrite) { status in
                continuation.resume(returning: status)
            }
        }
    }
}
