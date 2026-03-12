import { useState, useRef, useCallback, useEffect } from 'react';
import {
  Text,
  View,
  StyleSheet,
  TouchableOpacity,
  ScrollView,
  SafeAreaView,
  Image,
  Modal,
  FlatList,
  ActionSheetIOS,
  Platform,
  Alert,
} from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import * as MediaLibrary from 'expo-media-library';
import { FaceLibrary } from 'react-native-visage';
import type {
  Embedding,
  FaceRect,
  Album,
  MatchResult,
  MatchMode,
} from 'react-native-visage';

interface EnrolledFace {
  uri: string;
  imageWidth: number;
  imageHeight: number;
  faceRect: FaceRect;
  embedding: Embedding;
  label: string;
}

interface MatchedPhoto {
  assetId: string;
  matches: MatchResult[];
}

const FACE_THUMB_SIZE = 56;
const MATCH_THUMB_SIZE = 80;

function FaceThumb({ face }: { face: EnrolledFace }) {
  // Center the face in the thumbnail, with 1.5x padding for context
  const scale =
    FACE_THUMB_SIZE /
    (Math.max(face.faceRect.width, face.faceRect.height) * 1.5);
  const cx = face.faceRect.x + face.faceRect.width / 2;
  const cy = face.faceRect.y + face.faceRect.height / 2;
  const left = FACE_THUMB_SIZE / 2 - cx * scale;
  const top = FACE_THUMB_SIZE / 2 - cy * scale;

  return (
    <View style={styles.faceThumb}>
      <Image
        source={{ uri: face.uri }}
        style={{
          width: face.imageWidth * scale,
          height: face.imageHeight * scale,
          position: 'absolute',
          left,
          top,
        }}
      />
      <Text style={styles.faceThumbLabel}>{face.label}</Text>
    </View>
  );
}

function MatchThumb({ photo }: { photo: MatchedPhoto }) {
  const [uri, setUri] = useState<string | null>(null);
  const bestSim = Math.max(...photo.matches.map((m) => m.similarity));

  useEffect(() => {
    MediaLibrary.getAssetInfoAsync(photo.assetId)
      .then((info) => {
        setUri(info.localUri ?? null);
      })
      .catch(() => {
        /* ignore — thumbnail just stays blank */
      });
  }, [photo.assetId]);

  return (
    <View style={styles.matchThumb}>
      {uri ? (
        <Image source={{ uri }} style={styles.matchThumbImage} />
      ) : (
        <View style={[styles.matchThumbImage, styles.matchThumbPlaceholder]} />
      )}
      {photo.matches.length > 1 && (
        <View style={styles.matchBadge}>
          <Text style={styles.matchBadgeText}>{photo.matches.length}</Text>
        </View>
      )}
      <View style={styles.matchSimBar}>
        <Text style={styles.matchSimText}>{(bestSim * 100).toFixed(0)}%</Text>
      </View>
    </View>
  );
}

export default function App() {
  const [logs, setLogs] = useState<string[]>([]);
  const [enrolledFaces, setEnrolledFaces] = useState<EnrolledFace[]>([]);
  const [matchedPhotos, setMatchedPhotos] = useState<MatchedPhoto[]>([]);
  const [isScanning, setIsScanning] = useState(false);
  const [matchMode, setMatchMode] = useState<MatchMode>('any');
  const [albumModalVisible, setAlbumModalVisible] = useState(false);
  const [albums, setAlbums] = useState<Album[]>([]);
  const faceCounter = useRef(0);

  const log = useCallback((message: string) => {
    console.log(`[Visage] ${message}`);
    setLogs((prev) => [...prev, message]);
  }, []);

  // ── Pick photos ────────────────────────────────────────────────────────────

  const pickPhotos = useCallback(async () => {
    try {
      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ['images'],
        allowsMultipleSelection: true,
        quality: 1,
      });

      if (result.canceled || result.assets.length === 0) {
        log('Image picker cancelled');
        return;
      }

      log(`Extracting faces from ${result.assets.length} photo(s)...`);
      const newFaces: EnrolledFace[] = [];

      for (const asset of result.assets) {
        const detections = await FaceLibrary.extractEmbeddings(asset.uri);
        log(`  ${asset.fileName ?? asset.uri}: ${detections.length} face(s)`);

        for (const detection of detections) {
          faceCounter.current += 1;
          newFaces.push({
            uri: asset.uri,
            imageWidth: asset.width,
            imageHeight: asset.height,
            faceRect: detection.faceRect,
            embedding: detection.embedding,
            label: `F${faceCounter.current}`,
          });
        }
      }

      if (newFaces.length > 0) {
        setEnrolledFaces((prev) => [...prev, ...newFaces]);
        log(
          `Enrolled ${newFaces.length} face(s) — total: ${
            enrolledFaces.length + newFaces.length
          }`
        );
      } else {
        log('No faces detected in selected photos');
      }
    } catch (error) {
      log(`Error: ${error}`);
    }
  }, [log, enrolledFaces.length]);

  // ── Scan ───────────────────────────────────────────────────────────────────

  const startScan = useCallback(
    async (albumId?: string, albumName?: string) => {
      if (enrolledFaces.length === 0) {
        log('No enrolled faces — pick photos first');
        return;
      }

      setIsScanning(true);
      setMatchedPhotos([]);
      log(
        albumName
          ? `Scanning album "${albumName}"...`
          : 'Scanning full library...'
      );

      try {
        await FaceLibrary.scanLibrary({
          embeddings: enrolledFaces.map((f) => f.embedding),
          threshold: 0.65,
          matchMode,
          albumId,
          onProgress: (scanned, total) => {
            log(`Progress: ${scanned}/${total}`);
          },
          onMatch: (assetId, matches) => {
            log(
              `Match: ${matches
                .map(
                  (m) =>
                    `F${m.embeddingIndex + 1} ${(m.similarity * 100).toFixed(
                      1
                    )}%`
                )
                .join(', ')}`
            );
            setMatchedPhotos((prev) => [...prev, { assetId, matches }]);
          },
          onComplete: () => {
            log('Scan complete');
            setIsScanning(false);
          },
          onError: (error) => {
            log(`Scan error: ${error.message}`);
            setIsScanning(false);
          },
        });
      } catch (error) {
        log(`Scan failed: ${error}`);
        setIsScanning(false);
      }
    },
    [log, enrolledFaces, matchMode]
  );

  const cancelScan = useCallback(async () => {
    log('Cancelling scan...');
    await FaceLibrary.cancelScan();
  }, [log]);

  // ── Album picker ───────────────────────────────────────────────────────────

  const openAlbumPicker = useCallback(async () => {
    if (enrolledFaces.length === 0) {
      log('No enrolled faces — pick photos first');
      return;
    }

    try {
      log('Loading albums...');
      const fetchedAlbums = await FaceLibrary.getAlbums();
      log(`Found ${fetchedAlbums.length} album(s)`);

      if (fetchedAlbums.length === 0) {
        log('No albums found');
        return;
      }

      if (Platform.OS === 'ios') {
        ActionSheetIOS.showActionSheetWithOptions(
          {
            options: [
              'Cancel',
              ...fetchedAlbums.map((a) => `${a.name} (${a.count})`),
            ],
            cancelButtonIndex: 0,
            title: 'Select Album to Scan',
          },
          (buttonIndex) => {
            if (buttonIndex === 0) return;
            const album = fetchedAlbums[buttonIndex - 1]!;
            startScan(album.id, album.name);
          }
        );
      } else {
        setAlbums(fetchedAlbums);
        setAlbumModalVisible(true);
      }
    } catch (error) {
      log(`Failed to load albums: ${error}`);
    }
  }, [log, enrolledFaces.length, startScan]);

  const resetFaces = useCallback(() => {
    Alert.alert('Reset Faces', 'Remove all enrolled faces and matches?', [
      { text: 'Cancel', style: 'cancel' },
      {
        text: 'Reset',
        style: 'destructive',
        onPress: () => {
          setEnrolledFaces([]);
          setMatchedPhotos([]);
          faceCounter.current = 0;
          log('Enrolled faces cleared');
        },
      },
    ]);
  }, [log]);

  return (
    <SafeAreaView style={styles.container}>
      <Text style={styles.title}>Visage</Text>

      {/* Enrolled faces */}
      <View style={styles.section}>
        <View style={styles.sectionHeader}>
          <Text style={styles.sectionTitle}>
            Enrolled faces ({enrolledFaces.length})
          </Text>
          {enrolledFaces.length > 0 && (
            <TouchableOpacity onPress={resetFaces}>
              <Text style={styles.resetText}>Reset</Text>
            </TouchableOpacity>
          )}
        </View>
        {enrolledFaces.length === 0 ? (
          <Text style={styles.emptyHint}>Pick photos to enroll faces</Text>
        ) : (
          <ScrollView horizontal showsHorizontalScrollIndicator={false}>
            {enrolledFaces.map((face, i) => (
              <FaceThumb key={i} face={face} />
            ))}
          </ScrollView>
        )}
      </View>

      {/* Matched photos */}
      {matchedPhotos.length > 0 && (
        <View style={styles.section}>
          <View style={styles.sectionHeader}>
            <Text style={styles.sectionTitle}>
              Matches ({matchedPhotos.length})
            </Text>
          </View>
          <ScrollView horizontal showsHorizontalScrollIndicator={false}>
            {matchedPhotos.map((photo, i) => (
              <MatchThumb key={i} photo={photo} />
            ))}
          </ScrollView>
        </View>
      )}

      {/* Match mode toggle */}
      <View style={styles.modeToggle}>
        {(
          [
            { mode: 'any', label: 'Any' },
            { mode: 'all', label: 'All' },
            { mode: 'exact', label: 'Exact' },
          ] as const
        ).map(({ mode: m, label }) => (
          <TouchableOpacity
            key={m}
            style={[styles.modeBtn, matchMode === m && styles.modeBtnActive]}
            onPress={() => setMatchMode(m)}
          >
            <Text
              style={[
                styles.modeBtnText,
                matchMode === m && styles.modeBtnTextActive,
              ]}
            >
              {label}
            </Text>
          </TouchableOpacity>
        ))}
      </View>

      {/* Action buttons */}
      <View style={styles.buttons}>
        <TouchableOpacity
          style={styles.button}
          onPress={pickPhotos}
          disabled={isScanning}
        >
          <Text style={styles.buttonText}>Pick Photos</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[styles.button, enrolledFaces.length === 0 && styles.disabled]}
          onPress={() => startScan()}
          disabled={isScanning || enrolledFaces.length === 0}
        >
          <Text style={styles.buttonText}>Scan Library</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[styles.button, enrolledFaces.length === 0 && styles.disabled]}
          onPress={openAlbumPicker}
          disabled={isScanning || enrolledFaces.length === 0}
        >
          <Text style={styles.buttonText}>Scan Album</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[
            styles.button,
            styles.cancelButton,
            !isScanning && styles.disabled,
          ]}
          onPress={cancelScan}
          disabled={!isScanning}
        >
          <Text style={styles.buttonText}>Cancel</Text>
        </TouchableOpacity>
      </View>

      {/* Log output */}
      <ScrollView style={styles.logContainer}>
        {logs.map((entry, i) => (
          <Text key={i} style={styles.logText}>
            {entry}
          </Text>
        ))}
      </ScrollView>

      {/* Android album picker modal */}
      <Modal
        visible={albumModalVisible}
        animationType="slide"
        onRequestClose={() => setAlbumModalVisible(false)}
      >
        <SafeAreaView style={styles.modalContainer}>
          <View style={styles.modalHeader}>
            <Text style={styles.modalTitle}>Select Album</Text>
            <TouchableOpacity onPress={() => setAlbumModalVisible(false)}>
              <Text style={styles.modalClose}>Cancel</Text>
            </TouchableOpacity>
          </View>
          <FlatList
            data={albums}
            keyExtractor={(item) => item.id}
            renderItem={({ item }) => (
              <TouchableOpacity
                style={styles.albumRow}
                onPress={() => {
                  setAlbumModalVisible(false);
                  startScan(item.id, item.name);
                }}
              >
                <Text style={styles.albumName}>{item.name}</Text>
                <Text style={styles.albumCount}>{item.count} photos</Text>
              </TouchableOpacity>
            )}
          />
        </SafeAreaView>
      </Modal>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    textAlign: 'center',
    marginTop: 16,
    marginBottom: 12,
  },
  // Sections
  section: {
    paddingHorizontal: 16,
    marginBottom: 12,
  },
  sectionHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 8,
  },
  sectionTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#333',
  },
  resetText: {
    fontSize: 14,
    color: '#FF3B30',
    fontWeight: '600',
  },
  emptyHint: {
    fontSize: 13,
    color: '#999',
    fontStyle: 'italic',
  },
  // Enrolled face thumbnails
  faceThumb: {
    width: FACE_THUMB_SIZE,
    height: FACE_THUMB_SIZE,
    borderRadius: FACE_THUMB_SIZE / 2,
    overflow: 'hidden',
    marginRight: 8,
    backgroundColor: '#eee',
  },
  faceThumbLabel: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    backgroundColor: 'rgba(0,0,0,0.45)',
    color: '#fff',
    fontSize: 9,
    textAlign: 'center',
    fontWeight: '700',
  },
  // Match thumbnails
  matchThumb: {
    width: MATCH_THUMB_SIZE,
    height: MATCH_THUMB_SIZE,
    borderRadius: 8,
    overflow: 'hidden',
    marginRight: 8,
    backgroundColor: '#eee',
  },
  matchThumbImage: {
    width: MATCH_THUMB_SIZE,
    height: MATCH_THUMB_SIZE,
  },
  matchThumbPlaceholder: {
    backgroundColor: '#ddd',
  },
  matchBadge: {
    position: 'absolute',
    top: 4,
    right: 4,
    backgroundColor: '#007AFF',
    borderRadius: 8,
    minWidth: 16,
    paddingHorizontal: 3,
    alignItems: 'center',
  },
  matchBadgeText: {
    color: '#fff',
    fontSize: 10,
    fontWeight: '700',
  },
  matchSimBar: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    backgroundColor: 'rgba(0,0,0,0.5)',
    paddingVertical: 2,
    alignItems: 'center',
  },
  matchSimText: {
    color: '#fff',
    fontSize: 10,
    fontWeight: '700',
  },
  // Match mode toggle
  modeToggle: {
    flexDirection: 'row',
    justifyContent: 'center',
    gap: 8,
    paddingHorizontal: 16,
    marginBottom: 10,
  },
  modeBtn: {
    paddingHorizontal: 16,
    paddingVertical: 7,
    borderRadius: 20,
    borderWidth: 1,
    borderColor: '#007AFF',
  },
  modeBtnActive: {
    backgroundColor: '#007AFF',
  },
  modeBtnText: {
    fontSize: 13,
    fontWeight: '600',
    color: '#007AFF',
  },
  modeBtnTextActive: {
    color: '#fff',
  },
  // Buttons
  buttons: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'center',
    gap: 8,
    paddingHorizontal: 16,
    marginBottom: 12,
  },
  button: {
    backgroundColor: '#007AFF',
    paddingHorizontal: 16,
    paddingVertical: 10,
    borderRadius: 8,
  },
  cancelButton: {
    backgroundColor: '#FF3B30',
  },
  disabled: {
    opacity: 0.4,
  },
  buttonText: {
    color: '#fff',
    fontWeight: '600',
    fontSize: 14,
  },
  // Log
  logContainer: {
    flex: 1,
    backgroundColor: '#1a1a1a',
    marginHorizontal: 16,
    marginBottom: 16,
    borderRadius: 8,
    padding: 12,
  },
  logText: {
    color: '#0f0',
    fontFamily: 'Courier',
    fontSize: 12,
    marginBottom: 2,
  },
  // Album modal
  modalContainer: {
    flex: 1,
    backgroundColor: '#fff',
  },
  modalHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 16,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: '#ccc',
  },
  modalTitle: {
    fontSize: 18,
    fontWeight: '600',
  },
  modalClose: {
    fontSize: 16,
    color: '#007AFF',
  },
  albumRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 16,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: '#eee',
  },
  albumName: {
    fontSize: 16,
  },
  albumCount: {
    fontSize: 14,
    color: '#999',
  },
});
