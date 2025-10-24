import 'dart:typed_data';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:onnxruntime/onnxruntime.dart';

class EdgeSAMProcessor {
  late OrtSession _encoderSession;
  late OrtSession _decoderSession;
  bool _isInitialized = false;

  // Model constants
  static const int imageSize = 1024;
  static const int embedDim = 256;
  static const int embedSize = 64; // 1024/16 = 64

  // Preprocessing constants
  static const List<double> pixelMean = [123.675, 116.28, 103.53];
  static const List<double> pixelStd = [58.395, 57.12, 57.375];

  Future<void> initialize() async {
    if (_isInitialized) return;

    try {
      // Load encoder model
      final encoderData = await rootBundle.load(
        'assets/models/edge_sam_3x_encoder.onnx',
      );
      _encoderSession = OrtSession.fromBuffer(
        encoderData.buffer.asUint8List(),
        OrtSessionOptions(),
      );

      // Load decoder model
      final decoderData = await rootBundle.load(
        'assets/models/edge_sam_3x_decoder.onnx',
      );
      _decoderSession = OrtSession.fromBuffer(
        decoderData.buffer.asUint8List(),
        OrtSessionOptions(),
      );

      _isInitialized = true;
      debugPrint('EdgeSAM models loaded successfully');
    } catch (e) {
      debugPrint('Error initializing EdgeSAM: $e');
      rethrow;
    }
  }

  Future<Uint8List?> getImageEmbedding(Uint8List imageBytes) async {
    if (!_isInitialized) {
      await initialize();
    }

    try {
      // Decode and preprocess image
      final image = img.decodeImage(imageBytes);
      if (image == null) return null;

      final preprocessedImage = _preprocessImage(image);

      // Flatten the preprocessed image for ONNX tensor
      final flattenedImage = <double>[];
      for (final pixel in preprocessedImage) {
        flattenedImage.addAll(pixel); // Add R, G, B values
      }

      // Convert to Float32List for ONNX compatibility
      final flattenedImageFloat = Float32List.fromList(flattenedImage);

      // Run encoder inference
      final encoderInputs = {
        'image': OrtValueTensor.createTensorWithDataList(flattenedImageFloat, [
          1,
          3,
          imageSize,
          imageSize,
        ]),
      };

      final encoderOutputs = _encoderSession.run(
        OrtRunOptions(),
        encoderInputs,
      );
      final embeddings =
          encoderOutputs[0]?.value as List<List<List<List<double>>>>?;

      if (embeddings == null) return null;
      return Uint8List.fromList(
        _flattenEmbeddings(
          embeddings,
        ).map((e) => (e * 255).round().clamp(0, 255)).toList(),
      );
    } catch (e) {
      debugPrint('Error getting image embedding: $e');
      return null;
    }
  }

  Future<List<Uint8List>?> segmentImage(
    Uint8List imageBytes,
    List<Offset> points,
    List<int> labels,
  ) async {
    if (!_isInitialized) {
      await initialize();
    }

    try {
      // Get image embeddings
      final embeddings = await getImageEmbedding(imageBytes);
      if (embeddings == null) return null;

      // Get original image dimensions for coordinate scaling
      final originalImage = img.decodeImage(imageBytes);
      if (originalImage == null) return null;

      // Prepare point coordinates and labels
      final pointCoords = _preparePointCoords(
        points,
        originalImage.width,
        originalImage.height,
      );
      final pointLabels = labels.map((label) => label.toDouble()).toList();

      // Convert embeddings back to List<double> for ONNX
      final embeddingsList = embeddings.map((e) => e / 255.0).toList();

      // Convert to Float32List for ONNX compatibility
      final embeddingsListFloat = Float32List.fromList(embeddingsList);
      final pointCoordsFlat = <double>[];
      for (final coord in pointCoords) {
        pointCoordsFlat.addAll(coord);
      }
      final pointCoordsFloat = Float32List.fromList(pointCoordsFlat);
      final pointLabelsFloat = Float32List.fromList(pointLabels);

      // Run decoder inference
      final decoderInputs = {
        'image_embeddings': OrtValueTensor.createTensorWithDataList(
          embeddingsListFloat,
          [1, embedDim, embedSize, embedSize],
        ),
        'point_coords': OrtValueTensor.createTensorWithDataList(
          pointCoordsFloat,
          [1, points.length, 2],
        ),
        'point_labels': OrtValueTensor.createTensorWithDataList(
          pointLabelsFloat,
          [1, points.length],
        ),
      };

      final decoderOutputs = _decoderSession.run(
        OrtRunOptions(),
        decoderInputs,
      );
      final masks = decoderOutputs[1]?.value as List<List<List<List<double>>>>?;

      // Convert masks to images
      if (masks == null) return null;
      return _convertMasksToImages(masks, imageBytes);
    } catch (e) {
      debugPrint('Error segmenting image: $e');
      return null;
    }
  }

  List<List<double>> _preprocessImage(img.Image image) {
    // Resize image to 1024x1024
    final resized = img.copyResize(image, width: imageSize, height: imageSize);

    // Convert to RGB and normalize
    final List<List<double>> result = [];

    for (int y = 0; y < imageSize; y++) {
      for (int x = 0; x < imageSize; x++) {
        final pixel = resized.getPixel(x, y);
        final r = pixel.r / 255.0;
        final g = pixel.g / 255.0;
        final b = pixel.b / 255.0;

        // Normalize with mean and std
        final normalizedR = (r * 255.0 - pixelMean[0]) / pixelStd[0];
        final normalizedG = (g * 255.0 - pixelMean[1]) / pixelStd[1];
        final normalizedB = (b * 255.0 - pixelMean[2]) / pixelStd[2];

        result.add([normalizedR, normalizedG, normalizedB]);
      }
    }

    return result;
  }

  List<double> _flattenEmbeddings(List<List<List<List<double>>>> embeddings) {
    final List<double> result = [];
    for (int c = 0; c < embedDim; c++) {
      for (int h = 0; h < embedSize; h++) {
        for (int w = 0; w < embedSize; w++) {
          result.add(embeddings[0][c][h][w]);
        }
      }
    }
    return result;
  }

  List<List<double>> _preparePointCoords(
    List<Offset> points,
    int originalWidth,
    int originalHeight,
  ) {
    final List<List<double>> coords = [];
    debugPrint('Preparing point coordinates:');
    debugPrint('  Original image size: ${originalWidth}x${originalHeight}');
    for (final point in points) {
      debugPrint('  Original point: ${point.dx}, ${point.dy}');
      // Scale coordinates from original image size to 1024x1024 (model input size)
      final scaledX = (point.dx / originalWidth) * 1024.0;
      final scaledY = (point.dy / originalHeight) * 1024.0;
      coords.add([scaledX, scaledY]);
      debugPrint('  Scaled point: $scaledX, $scaledY');
    }
    debugPrint('  Final coords: $coords');
    return coords;
  }

  List<Uint8List> _convertMasksToImages(
    List<List<List<List<double>>>> masks,
    Uint8List originalImageBytes,
  ) {
    final List<Uint8List> maskImages = [];
    final originalImage = img.decodeImage(originalImageBytes);
    if (originalImage == null) return maskImages;

    for (final mask in masks[0]) {
      // Get mask dimensions (should be 256x256 from model output)
      final maskHeight = mask.length;
      final maskWidth = mask[0].length;

      debugPrint('Mask dimensions: ${maskWidth}x${maskHeight}');
      debugPrint(
        'Original image dimensions: ${originalImage.width}x${originalImage.height}',
      );

      // Debug: Check mask value range
      double minVal = 1.0;
      double maxVal = 0.0;
      int nonZeroCount = 0;
      for (int y = 0; y < maskHeight; y++) {
        for (int x = 0; x < maskWidth; x++) {
          final val = mask[y][x];
          if (val > 0) nonZeroCount++;
          if (val < minVal) minVal = val;
          if (val > maxVal) maxVal = val;
        }
      }
      debugPrint('Mask value range: $minVal to $maxVal');
      debugPrint(
        'Non-zero pixels: $nonZeroCount out of ${maskWidth * maskHeight}',
      );

      // Create mask image with original dimensions
      final maskImage = img.Image(
        width: originalImage.width,
        height: originalImage.height,
      );

      for (int y = 0; y < originalImage.height; y++) {
        for (int x = 0; x < originalImage.width; x++) {
          // Scale coordinates to mask dimensions
          final maskX = (x * maskWidth / originalImage.width).round().clamp(
            0,
            maskWidth - 1,
          );
          final maskY = (y * maskHeight / originalImage.height).round().clamp(
            0,
            maskHeight - 1,
          );

          // Get mask value (0-1)
          final maskValue = mask[maskY][maskX];

          // Only show pixels where mask value is above threshold (e.g., 0.5)
          if (maskValue > 0.5) {
            // Create red segmentation mask
            maskImage.setPixel(x, y, img.ColorRgba8(255, 0, 0, 180));
          } else {
            // Transparent for non-segmented areas
            maskImage.setPixel(x, y, img.ColorRgba8(0, 0, 0, 0));
          }
        }
      }

      maskImages.add(Uint8List.fromList(img.encodePng(maskImage)));
    }

    return maskImages;
  }

  void dispose() {
    _encoderSession.release();
    _decoderSession.release();
    _isInitialized = false;
  }
}
