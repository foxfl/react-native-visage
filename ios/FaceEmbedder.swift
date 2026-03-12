import CoreML
import Accelerate

class FaceEmbedder {

    // MARK: - Types

    enum NormalizationMode {
        /// Normalize pixels from [0, 255] to [-1, 1]: value / 127.5 - 1.0
        case negOneOne
        /// Normalize pixels from [0, 255] to [0, 1]: value / 255.0
        case zeroOne
    }

    // MARK: - Properties

    private let model: MLModel?
    let embeddingSize: Int
    private let inputSize: Int
    private let inputLayerName: String
    private let outputLayerName: String
    private let normalizationMode: NormalizationMode

    // MARK: - Init

    /// Loads the bundled InceptionResNetV1 model (default — works out of the box).
    /// Trained on VGGFace2, robust to diverse poses and partial faces.
    convenience init() {
        let url = Bundle(for: FaceEmbedder.self).url(
            forResource: "MobileFaceNet", withExtension: "mlmodelc"
        )
        self.init(
            modelURL: url,
            embeddingSize: 512,
            inputSize: 160,
            outputLayerName: nil,
            normalization: .negOneOne
        )
    }

    /// Loads a custom CoreML model from the given URL.
    /// - Parameters:
    ///   - modelURL: Path to a compiled `.mlmodelc` directory.
    ///   - embeddingSize: Expected output embedding dimension (e.g. 512).
    ///   - inputSize: Square input side length the model expects (e.g. 112).
    ///   - outputLayerName: Name of the output feature to use as the embedding.
    ///                      Pass `nil` to auto-detect from the model's first output.
    ///   - normalization: Pixel normalization applied before inference.
    init(modelURL: URL?, embeddingSize: Int, inputSize: Int,
         outputLayerName: String?, normalization: NormalizationMode) {
        self.embeddingSize = embeddingSize
        self.inputSize = inputSize
        self.normalizationMode = normalization

        if let url = modelURL, let m = try? MLModel(contentsOf: url) {
            self.model = m
            self.inputLayerName = m.modelDescription.inputDescriptionsByName.keys.first ?? "input_1"
            self.outputLayerName = outputLayerName
                ?? m.modelDescription.outputDescriptionsByName.keys.first
                ?? "output"
        } else {
            self.model = nil
            self.inputLayerName = "input_1"
            self.outputLayerName = outputLayerName ?? "output"
        }
    }

    // MARK: - Embedding extraction

    /// Extract an L2-normalized embedding from a cropped face image.
    /// Returns a zeroed vector if the model is not loaded (stub mode).
    func getEmbedding(from faceImage: CGImage) throws -> [Float] {
        guard let model = model else {
            return [Float](repeating: 0.0, count: embeddingSize)
        }

        let multiArray = try createInputArray(from: faceImage)
        let input = try MLDictionaryFeatureProvider(dictionary: [
            inputLayerName: MLFeatureValue(multiArray: multiArray),
        ])

        let output = try model.prediction(from: input)

        guard let outputFeature = output.featureValue(for: outputLayerName),
              let outputArray = outputFeature.multiArrayValue else {
            throw VisageError.modelLoadFailed
        }

        let count = outputArray.count
        guard count == embeddingSize else {
            throw VisageError.embeddingDimensionMismatch(
                expected: embeddingSize, got: count
            )
        }

        var embedding = [Float](repeating: 0.0, count: count)
        let ptr = outputArray.dataPointer.bindMemory(to: Float.self, capacity: count)
        for i in 0..<count {
            embedding[i] = ptr[i]
        }

        // L2-normalize
        var norm: Float = 0
        vDSP_svesq(embedding, 1, &norm, vDSP_Length(count))
        norm = sqrt(norm)
        if norm > 0 {
            var divisor = norm
            vDSP_vsdiv(embedding, 1, &divisor, &embedding, 1, vDSP_Length(count))
        }

        return embedding
    }

    // MARK: - Private helpers

    /// Resize the face crop to `inputSize × inputSize` and create an MLMultiArray
    /// in NCHW format [1, 3, H, W] with pixels normalized per `normalizationMode`.
    private func createInputArray(from image: CGImage) throws -> MLMultiArray {
        let size = inputSize
        let bytesPerRow = size * 4
        var pixelData = [UInt8](repeating: 0, count: size * size * 4)

        guard let context = CGContext(
            data: &pixelData,
            width: size,
            height: size,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
        ) else {
            throw VisageError.modelLoadFailed
        }
        context.draw(image, in: CGRect(x: 0, y: 0, width: size, height: size))

        let array = try MLMultiArray(
            shape: [1, 3, NSNumber(value: size), NSNumber(value: size)],
            dataType: .float32
        )
        let arrayPtr = array.dataPointer.bindMemory(
            to: Float.self, capacity: 3 * size * size
        )

        // Convert interleaved RGBA [H, W, 4] → planar RGB [3, H, W] with normalization
        for y in 0..<size {
            for x in 0..<size {
                let srcIdx = (y * size + x) * 4
                let dstBase = y * size + x
                for c in 0..<3 {
                    let raw = Float(pixelData[srcIdx + c])
                    arrayPtr[c * size * size + dstBase] = normalizationMode == .negOneOne
                        ? raw / 127.5 - 1.0
                        : raw / 255.0
                }
            }
        }

        return array
    }
}
