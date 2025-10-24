import 'dart:typed_data';
import 'dart:ui' as ui;
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:permission_handler/permission_handler.dart';
import 'edgesam_processor.dart';

void main() {
  runApp(const EdgeSAMDemoApp());
}

class EdgeSAMDemoApp extends StatelessWidget {
  const EdgeSAMDemoApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'EdgeSAM Demo',
      theme: ThemeData(primarySwatch: Colors.blue, useMaterial3: true),
      home: const EdgeSAMScreen(),
    );
  }
}

class EdgeSAMScreen extends StatefulWidget {
  const EdgeSAMScreen({super.key});

  @override
  State<EdgeSAMScreen> createState() => _EdgeSAMScreenState();
}

class _EdgeSAMScreenState extends State<EdgeSAMScreen> {
  final EdgeSAMProcessor _processor = EdgeSAMProcessor();
  final ImagePicker _picker = ImagePicker();

  Uint8List? _imageBytes;
  ui.Image? _displayImage;
  final List<Offset> _points = [];
  final List<int> _labels = []; // 1 for positive, 0 for negative
  List<Uint8List>? _masks;
  List<ui.Image>? _decodedMasks;
  bool _isProcessing = false;
  bool _isInitialized = false;
  bool _showMasks = true;

  @override
  void initState() {
    super.initState();
    _initializeProcessor();
  }

  Future<void> _initializeProcessor() async {
    try {
      await _processor.initialize();
      setState(() {
        _isInitialized = true;
      });
    } catch (e) {
      _showErrorDialog('Failed to initialize EdgeSAM: $e');
    }
  }

  Future<void> _pickImage() async {
    try {
      final XFile? image = await _picker.pickImage(
        source: ImageSource.gallery,
        maxWidth: 1024,
        maxHeight: 1024,
        imageQuality: 85,
      );

      if (image != null) {
        final bytes = await image.readAsBytes();
        final displayImage = await _bytesToImage(bytes);

        setState(() {
          _imageBytes = bytes;
          _displayImage = displayImage;
          _points.clear();
          _labels.clear();
          _masks = null;
        });
      }
    } catch (e) {
      _showErrorDialog('Failed to pick image: $e');
    }
  }

  Future<void> _takePhoto() async {
    try {
      // Request camera permission
      final status = await Permission.camera.request();
      if (!status.isGranted) {
        _showErrorDialog('Camera permission denied');
        return;
      }

      final XFile? image = await _picker.pickImage(
        source: ImageSource.camera,
        maxWidth: 1024,
        maxHeight: 1024,
        imageQuality: 85,
      );

      if (image != null) {
        final bytes = await image.readAsBytes();
        final displayImage = await _bytesToImage(bytes);

        setState(() {
          _imageBytes = bytes;
          _displayImage = displayImage;
          _points.clear();
          _labels.clear();
          _masks = null;
        });
      }
    } catch (e) {
      _showErrorDialog('Failed to take photo: $e');
    }
  }

  Future<ui.Image> _bytesToImage(Uint8List bytes) async {
    final codec = await ui.instantiateImageCodec(bytes);
    final frame = await codec.getNextFrame();
    return frame.image;
  }

  void _onImageTap(TapDownDetails details) {
    if (_displayImage == null) return;

    final RenderBox renderBox = context.findRenderObject() as RenderBox;
    final localPosition = renderBox.globalToLocal(details.globalPosition);

    // Convert tap position to image coordinates
    final imageSize = _getImageDisplaySize();
    final imageOffset = _getImageOffset(imageSize);

    final x =
        (localPosition.dx - imageOffset.dx) /
        imageSize.width *
        _displayImage!.width;
    final y =
        (localPosition.dy - imageOffset.dy) /
        imageSize.height *
        _displayImage!.height;

    if (x >= 0 &&
        x < _displayImage!.width &&
        y >= 0 &&
        y < _displayImage!.height) {
      debugPrint('Tap coordinates:');
      debugPrint('  Local position: ${localPosition.dx}, ${localPosition.dy}');
      debugPrint(
        '  Image size: ${_displayImage!.width}x${_displayImage!.height}',
      );
      debugPrint('  Display size: ${imageSize.width}x${imageSize.height}');
      debugPrint('  Image offset: ${imageOffset.dx}, ${imageOffset.dy}');
      debugPrint('  Calculated point: $x, $y');

      setState(() {
        _points.add(Offset(x, y));
        _labels.add(1); // Default to positive point
      });

      _segmentImage();
    }
  }

  Future<void> _segmentImage() async {
    if (_imageBytes == null || _points.isEmpty || _isProcessing) return;

    setState(() {
      _isProcessing = true;
    });

    try {
      final masks = await _processor.segmentImage(
        _imageBytes!,
        _points,
        _labels,
      );

      // Decode mask images for display
      List<ui.Image>? decodedMasks;
      if (masks != null) {
        decodedMasks = [];
        for (final maskBytes in masks) {
          try {
            final codec = await ui.instantiateImageCodec(maskBytes);
            final frame = await codec.getNextFrame();
            decodedMasks.add(frame.image);
          } catch (e) {
            debugPrint('Error decoding mask: $e');
          }
        }
      }

      setState(() {
        _masks = masks;
        _decodedMasks = decodedMasks;
        _isProcessing = false;
      });
    } catch (e) {
      setState(() {
        _isProcessing = false;
      });
      _showErrorDialog('Segmentation failed: $e');
    }
  }

  void _clearPoints() {
    setState(() {
      _points.clear();
      _labels.clear();
      _masks = null;
      _decodedMasks = null;
    });
  }

  void _showErrorDialog(String message) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Error'),
        content: Text(message),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(context).pop(),
            child: const Text('OK'),
          ),
        ],
      ),
    );
  }

  Size _getImageDisplaySize() {
    if (_displayImage == null) return Size.zero;

    final screenSize = MediaQuery.of(context).size;
    final imageAspectRatio = _displayImage!.width / _displayImage!.height;
    final screenAspectRatio = screenSize.width / screenSize.height;

    if (imageAspectRatio > screenAspectRatio) {
      return Size(screenSize.width, screenSize.width / imageAspectRatio);
    } else {
      return Size(screenSize.height * imageAspectRatio, screenSize.height);
    }
  }

  Offset _getImageOffset(Size imageSize) {
    final screenSize = MediaQuery.of(context).size;
    return Offset(
      (screenSize.width - imageSize.width) / 2,
      (screenSize.height - imageSize.height) / 2,
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('EdgeSAM Demo'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        actions: [
          IconButton(
            icon: const Icon(Icons.photo_library),
            onPressed: _pickImage,
            tooltip: 'Pick from Gallery',
          ),
          IconButton(
            icon: const Icon(Icons.camera_alt),
            onPressed: _takePhoto,
            tooltip: 'Take Photo',
          ),
          IconButton(
            icon: const Icon(Icons.clear),
            onPressed: _clearPoints,
            tooltip: 'Clear Points',
          ),
        ],
      ),
      body: _buildBody(),
      floatingActionButton: _buildFloatingActionButton(),
    );
  }

  Widget _buildBody() {
    if (!_isInitialized) {
      return const Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            CircularProgressIndicator(),
            SizedBox(height: 16),
            Text('Initializing EdgeSAM...'),
          ],
        ),
      );
    }

    if (_displayImage == null) {
      return const Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(Icons.image, size: 64, color: Colors.grey),
            SizedBox(height: 16),
            Text('Select an image to start segmentation'),
            SizedBox(height: 8),
            Text('Tap to add points, tap points to toggle positive/negative'),
          ],
        ),
      );
    }

    return Stack(
      children: [
        // Display image
        Center(
          child: GestureDetector(
            onTapDown: _onImageTap,
            child: CustomPaint(
              painter: ImagePainter(
                image: _displayImage!,
                points: _points,
                labels: _labels,
                masks: _masks,
                decodedMasks: _decodedMasks,
                showMasks: _showMasks,
              ),
              size: _getImageDisplaySize(),
            ),
          ),
        ),

        // Processing indicator
        if (_isProcessing)
          Container(
            color: Colors.black54,
            child: const Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  CircularProgressIndicator(color: Colors.white),
                  SizedBox(height: 16),
                  Text(
                    'Processing...',
                    style: TextStyle(color: Colors.white, fontSize: 18),
                  ),
                ],
              ),
            ),
          ),
      ],
    );
  }

  Widget? _buildFloatingActionButton() {
    if (_displayImage == null) return null;

    return Row(
      mainAxisAlignment: MainAxisAlignment.end,
      children: [
        // Toggle masks button
        if (_masks != null)
          FloatingActionButton(
            onPressed: () {
              setState(() {
                _showMasks = !_showMasks;
              });
            },
            tooltip: _showMasks ? 'Hide Masks' : 'Show Masks',
            backgroundColor: _showMasks ? Colors.red : Colors.grey,
            child: Icon(_showMasks ? Icons.visibility_off : Icons.visibility),
          ),
        const SizedBox(width: 16),
        // Segment button
        FloatingActionButton(
          onPressed: _segmentImage,
          tooltip: 'Segment Image',
          child: const Icon(Icons.segment),
        ),
      ],
    );
  }

  @override
  void dispose() {
    _processor.dispose();
    super.dispose();
  }
}

class ImagePainter extends CustomPainter {
  final ui.Image image;
  final List<Offset> points;
  final List<int> labels;
  final List<Uint8List>? masks;
  final List<ui.Image>? decodedMasks;
  final bool showMasks;

  ImagePainter({
    required this.image,
    required this.points,
    required this.labels,
    this.masks,
    this.decodedMasks,
    this.showMasks = true,
  });

  @override
  void paint(Canvas canvas, Size size) {
    // Draw image
    final paint = Paint();
    final srcRect = Rect.fromLTWH(
      0,
      0,
      image.width.toDouble(),
      image.height.toDouble(),
    );
    final dstRect = Rect.fromLTWH(0, 0, size.width, size.height);
    canvas.drawImageRect(image, srcRect, dstRect, paint);

    // Draw masks
    if (showMasks && decodedMasks != null && decodedMasks!.isNotEmpty) {
      for (int i = 0; i < decodedMasks!.length; i++) {
        final maskImage = decodedMasks![i];

        // Create a paint for mask overlay
        final maskPaint = Paint()..blendMode = BlendMode.srcATop;

        // Scale mask to display size
        final maskSrcRect = Rect.fromLTWH(
          0,
          0,
          maskImage.width.toDouble(),
          maskImage.height.toDouble(),
        );
        final maskDstRect = Rect.fromLTWH(0, 0, size.width, size.height);

        // Draw the mask overlay
        canvas.drawImageRect(maskImage, maskSrcRect, maskDstRect, maskPaint);
      }
    } else if (showMasks && masks != null && masks!.isNotEmpty) {
      // Fallback: draw a simple overlay to show segmentation is working
      final maskPaint = Paint()
        ..color = Colors.red.withOpacity(0.3)
        ..style = PaintingStyle.fill;

      // Draw a semi-transparent overlay to indicate segmentation
      canvas.drawRect(Rect.fromLTWH(0, 0, size.width, size.height), maskPaint);

      // Draw mask boundary indicators
      final boundaryPaint = Paint()
        ..color = Colors.red
        ..style = PaintingStyle.stroke
        ..strokeWidth = 3;

      // Draw a border around the image to show segmentation area
      canvas.drawRect(
        Rect.fromLTWH(0, 0, size.width, size.height),
        boundaryPaint,
      );
    }

    // Draw points
    for (int i = 0; i < points.length; i++) {
      final point = points[i];
      final label = labels[i];

      // Scale point to display size
      final displayX = point.dx / image.width * size.width;
      final displayY = point.dy / image.height * size.height;

      final pointPaint = Paint()
        ..color = label == 1 ? Colors.green : Colors.red
        ..style = PaintingStyle.fill;

      canvas.drawCircle(Offset(displayX, displayY), 8, pointPaint);

      // Draw border
      final borderPaint = Paint()
        ..color = Colors.white
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2;

      canvas.drawCircle(Offset(displayX, displayY), 8, borderPaint);
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) {
    return oldDelegate is ImagePainter &&
        (oldDelegate.points != points ||
            oldDelegate.labels != labels ||
            oldDelegate.masks != masks ||
            oldDelegate.decodedMasks != decodedMasks ||
            oldDelegate.showMasks != showMasks);
  }
}
