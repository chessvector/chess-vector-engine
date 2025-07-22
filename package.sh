#!/bin/bash
# Chess Vector Engine - Distribution Package Creator
# Prepares binaries and documentation for release

set -e

VERSION="0.5.0"
PACKAGE_NAME="chess-vector-engine-v${VERSION}"

echo "🚀 Creating Chess Vector Engine v${VERSION} distribution package..."

# Create clean build
echo "📦 Building release binaries..."
cargo build --release

# Create package directory
PACKAGE_DIR="dist/${PACKAGE_NAME}"
mkdir -p "${PACKAGE_DIR}"

# Copy main binary
echo "📋 Copying UCI engine binary..."
cp target/release/uci_engine "${PACKAGE_DIR}/chess-vector-engine"
chmod +x "${PACKAGE_DIR}/chess-vector-engine"

# Copy documentation
echo "📚 Copying documentation..."
cp README.md "${PACKAGE_DIR}/"
cp INSTALL.md "${PACKAGE_DIR}/"
cp CLAUDE.md "${PACKAGE_DIR}/"
cp LICENSE "${PACKAGE_DIR}/"

# Copy sample data
echo "🎯 Copying sample training data..."
cp -r training_data "${PACKAGE_DIR}/"

# Create version info
echo "ℹ️ Creating version information..."
cat > "${PACKAGE_DIR}/VERSION.txt" << EOF
Chess Vector Engine v${VERSION} - "Validated Performance"
Built: $(date)
Platform: $(uname -s) $(uname -m)
Binary: chess-vector-engine (UCI)
Strength: 1600-1800 ELO (validated)

Key Features:
- UCI compliant chess engine
- Vector-based pattern recognition
- Tactical search with calibrated evaluation
- 90.9% tactical accuracy on validation tests
- 62.5% agreement with controlled Stockfish testing
- Fast startup (~1ms) and low memory usage (~70MB)

Installation:
1. Make binary executable: chmod +x chess-vector-engine
2. Add to your chess GUI as UCI engine
3. See INSTALL.md for detailed setup instructions

Support:
- GitHub: https://github.com/chessvector/chess-vector-engine
- Documentation: https://docs.rs/chess-vector-engine
EOF

# Create quick start script
echo "⚡ Creating quick start script..."
cat > "${PACKAGE_DIR}/test-engine.sh" << 'EOF'
#!/bin/bash
echo "🎯 Testing Chess Vector Engine..."
echo "Sending UCI commands to verify engine functionality..."

echo -e "uci\nposition startpos\ngo movetime 1000\nquit" | ./chess-vector-engine

echo ""
echo "✅ Test complete! If you see 'bestmove' output above, the engine is working correctly."
echo "📚 See INSTALL.md for chess GUI setup instructions."
EOF

chmod +x "${PACKAGE_DIR}/test-engine.sh"

# Show package contents
echo ""
echo "📦 Package contents:"
ls -la "${PACKAGE_DIR}/"

# Calculate sizes
echo ""
echo "📊 Package statistics:"
echo "Binary size: $(ls -lh "${PACKAGE_DIR}/chess-vector-engine" | awk '{print $5}')"
echo "Total package size: $(du -sh "${PACKAGE_DIR}" | awk '{print $1}')"

# Test the packaged binary
echo ""
echo "🧪 Testing packaged binary..."
if "${PACKAGE_DIR}/chess-vector-engine" <<< "uci" | grep -q "uciok"; then
    echo "✅ Binary test successful - UCI protocol working"
else
    echo "❌ Binary test failed - UCI protocol issue"
    exit 1
fi

echo ""
echo "🎉 Package created successfully: ${PACKAGE_DIR}"
echo "📦 Ready for distribution!"

# Optional: Create archive
echo ""
read -p "Create .tar.gz archive? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "📦 Creating archive..."
    cd dist
    tar -czf "${PACKAGE_NAME}.tar.gz" "${PACKAGE_NAME}"
    echo "✅ Archive created: dist/${PACKAGE_NAME}.tar.gz"
    echo "📊 Archive size: $(ls -lh "${PACKAGE_NAME}.tar.gz" | awk '{print $5}')"
fi

echo ""
echo "🚀 Distribution package ready!"
echo "📍 Location: ${PACKAGE_DIR}"
EOF