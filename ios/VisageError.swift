import Foundation

enum VisageError: LocalizedError {
    case modelLoadFailed
    case photoAccessDenied
    case invalidImageUri(String)
    case embeddingDimensionMismatch(expected: Int, got: Int)

    var errorDescription: String? {
        switch self {
        case .modelLoadFailed:
            return "Failed to load face embedding model"
        case .photoAccessDenied:
            return "Photo library access denied"
        case .invalidImageUri(let uri):
            return "Failed to load image from URI: \(uri)"
        case .embeddingDimensionMismatch(let expected, let got):
            return "Embedding dimension mismatch: expected \(expected), got \(got)"
        }
    }
}
