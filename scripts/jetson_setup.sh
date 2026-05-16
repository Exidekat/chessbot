#!/bin/bash
# =============================================================================
# ChessBot Jetson Orin Nano Setup Script
# =============================================================================
# For JetPack 6.2.x with CUDA 12.6
#
# This script installs:
#   - System dependencies (v4l-utils, stockfish, build tools)
#   - cuSPARSELt 0.7.1 (required for PyTorch 24.06+)
#   - PyTorch 2.5.0 with CUDA support (NVIDIA Jetson wheel)
#   - torchvision 0.20.0 (built from source)
#
# Usage:
#   chmod +x scripts/jetson_setup.sh
#   ./scripts/jetson_setup.sh
#
# =============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
echo_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
echo_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Save project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# =============================================================================
# Pre-flight checks
# =============================================================================

echo "============================================================"
echo "ChessBot Jetson Orin Nano Setup"
echo "============================================================"
echo ""

# Check we're on a Jetson
if [ ! -f /etc/nv_tegra_release ]; then
    echo_error "This script is intended for NVIDIA Jetson devices only."
    exit 1
fi

echo_info "Detected Jetson device:"
cat /etc/nv_tegra_release
echo ""

# Check for conda environment
REQUIRED_PY_MAJOR=3
REQUIRED_PY_MINOR=10

if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo_error "No conda environment active."
    echo_error "The Jetson PyTorch wheel is cp310-only; installing into the system Python is unsupported."
    echo "    conda activate cb   # must be Python ${REQUIRED_PY_MAJOR}.${REQUIRED_PY_MINOR}"
    exit 1
fi

if [ "$CONDA_DEFAULT_ENV" = "base" ]; then
    echo_error "Refusing to install into the 'base' conda environment."
    echo_error "Project policy (CLAUDE.md) requires the 'cb' environment."
    if conda env list 2>/dev/null | awk '{print $1}' | grep -qx cb; then
        echo "    conda activate cb"
    else
        echo_warn "'cb' environment not found. Create it with:"
        echo "    conda create -n cb python=${REQUIRED_PY_MAJOR}.${REQUIRED_PY_MINOR} -y"
        echo "    conda activate cb"
    fi
    exit 1
fi

echo_info "Conda environment: $CONDA_DEFAULT_ENV"

# Assert Python version matches the pinned Jetson PyTorch wheel (cp310)
PY_MAJOR=$(python -c 'import sys; print(sys.version_info.major)')
PY_MINOR=$(python -c 'import sys; print(sys.version_info.minor)')
if [ "$PY_MAJOR" != "$REQUIRED_PY_MAJOR" ] || [ "$PY_MINOR" != "$REQUIRED_PY_MINOR" ]; then
    echo_error "Python ${PY_MAJOR}.${PY_MINOR} detected; this script requires Python ${REQUIRED_PY_MAJOR}.${REQUIRED_PY_MINOR}."
    echo_error "The NVIDIA Jetson PyTorch wheel is cp310-only and will not install on other versions."
    echo_warn "Recreate the environment with the correct Python:"
    echo "    conda deactivate"
    echo "    conda remove -n ${CONDA_DEFAULT_ENV} --all -y   # if you want to recreate"
    echo "    conda create -n cb python=${REQUIRED_PY_MAJOR}.${REQUIRED_PY_MINOR} -y"
    echo "    conda activate cb"
    exit 1
fi
echo_info "Python version: ${PY_MAJOR}.${PY_MINOR} (matches pinned cp310 wheel)"

# =============================================================================
# Step 1: System Dependencies
# =============================================================================

echo ""
echo "============================================================"
echo "Step 1: Installing System Dependencies"
echo "============================================================"

sudo apt-get update
sudo apt-get install -y \
    v4l-utils \
    stockfish \
    libjpeg-dev \
    zlib1g-dev \
    libpython3-dev \
    libopenblas-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    build-essential \
    git \
    ffmpeg \
    cmake
sudo apt-get install -y v4l2*

echo_info "System dependencies installed."

# Verify v4l2-ctl
if command -v v4l2-ctl &> /dev/null; then
    echo_info "v4l2-ctl installed: $(v4l2-ctl --version 2>&1 | head -1)"
else
    echo_error "v4l2-ctl installation failed"
fi

# Verify stockfish
if command -v stockfish &> /dev/null; then
    echo_info "Stockfish installed."
else
    echo_error "Stockfish installation failed"
fi

# =============================================================================
# Step 1.5: Build nvtop from source (apt version lacks Tegra support)
# =============================================================================

echo ""
echo "============================================================"
echo "Step 1.5: Installing jtop (Jetson GPU monitor)"
echo "============================================================"

# jtop (jetson-stats) is the proper monitoring tool for Jetson/Tegra devices
# nvtop doesn't support Tegra GPUs properly
if command -v jtop &> /dev/null; then
    echo_info "jtop already installed."
else
    echo_info "Installing jetson-stats (includes jtop)..."
    sudo pip3 install jetson-stats
    sudo systemctl restart jtop.service 2>/dev/null || true
    echo_info "jtop installed. You may need to reboot or re-login for full functionality."
fi

# Verify jtop
if command -v jtop &> /dev/null; then
    echo_info "jtop installed successfully."
else
    echo_warn "jtop installation may have failed - check manually"
fi

# =============================================================================
# Step 2: cuSPARSELt Installation
# =============================================================================

echo ""
echo "============================================================"
echo "Step 2: Installing cuSPARSELt 0.7.1"
echo "============================================================"

CUSPARSELT_DEB="cusparselt-local-tegra-repo-ubuntu2204-0.7.1_1.0-1_arm64.deb"
CUSPARSELT_URL="https://developer.download.nvidia.com/compute/cusparselt/0.7.1/local_installers/${CUSPARSELT_DEB}"

# Check if already installed
if dpkg -l | grep -q libcusparselt0; then
    echo_info "cuSPARSELt already installed, skipping."
else
    cd /tmp

    if [ ! -f "$CUSPARSELT_DEB" ]; then
        echo_info "Downloading cuSPARSELt..."
        wget -q --show-progress "$CUSPARSELT_URL"
    fi

    echo_info "Installing cuSPARSELt..."
    sudo dpkg -i "$CUSPARSELT_DEB"
    sudo cp /var/cusparselt-local-tegra-repo-ubuntu2204-0.7.1/cusparselt-*-keyring.gpg /usr/share/keyrings/
    sudo apt-get update
    sudo apt-get install -y libcusparselt0 libcusparselt-dev

    echo_info "cuSPARSELt installed."
fi

# Step 2.5: Other pip dependencies
python -m pip install -r "$PROJECT_ROOT/requirements.txt"

# =============================================================================
# Step 3: PyTorch Installation
# =============================================================================

echo ""
echo "============================================================"
echo "Step 3: Installing PyTorch 2.5.0 (NVIDIA Jetson Wheel)"
echo "============================================================"

PYTORCH_WHEEL="https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl"

# Check current PyTorch
CURRENT_TORCH=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "not installed")
echo_info "Current PyTorch: $CURRENT_TORCH"

if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo_info "PyTorch with CUDA already working, skipping."
else
    echo_info "Installing PyTorch with CUDA support..."

    # Uninstall existing PyTorch
    pip uninstall torch torchvision torchaudio -y 2>/dev/null || true

    # Install numpy (specific version required)
    pip install numpy==1.26.1

    # Install NVIDIA's Jetson PyTorch wheel
    pip install --no-cache "$PYTORCH_WHEEL"

    echo_info "PyTorch installed."
fi

# Verify PyTorch CUDA
echo_info "Verifying PyTorch CUDA support..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'Device: {torch.cuda.get_device_name(0)}')
"

# =============================================================================
# Step 4: Torchvision Installation (Build from Source)
# =============================================================================

echo ""
echo "============================================================"
echo "Step 4: Installing torchvision 0.20.0 (from source)"
echo "============================================================"

# Check if torchvision already works
if python3 -c "import torchvision; print(torchvision.__version__)" 2>/dev/null; then
    CURRENT_TV=$(python3 -c "import torchvision; print(torchvision.__version__)")
    echo_info "torchvision $CURRENT_TV already installed."
    read -p "Reinstall from source? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo_info "Skipping torchvision build."
        SKIP_TORCHVISION=1
    fi
fi

if [ -z "$SKIP_TORCHVISION" ]; then
    echo_info "Building torchvision from source (this may take 10-20 minutes)..."

    cd /tmp
    rm -rf torchvision

    git clone --branch release/0.20 --depth 1 https://github.com/pytorch/vision torchvision
    cd torchvision

    export BUILD_VERSION=0.20.0
    pip install --no-build-isolation .

    # Clean up build directory to prevent import conflicts
    cd /tmp
    rm -rf torchvision

    echo_info "torchvision installed."
fi

# =============================================================================
# Step 5: Verification
# =============================================================================

echo ""
echo "============================================================"
echo "Step 5: Final Verification"
echo "============================================================"

python3 << 'EOF'
import sys

def check_import(name):
    try:
        module = __import__(name)
        version = getattr(module, '__version__', 'unknown')
        return True, version
    except ImportError as e:
        return False, str(e)

print("Package Status:")
print("-" * 50)

# Check torch
ok, ver = check_import('torch')
print(f"  torch: {'✓' if ok else '✗'} {ver}")

if ok:
    import torch
    cuda_ok = torch.cuda.is_available()
    print(f"  torch.cuda: {'✓' if cuda_ok else '✗'} {'available' if cuda_ok else 'NOT AVAILABLE'}")

# Check torchvision
ok, ver = check_import('torchvision')
print(f"  torchvision: {'✓' if ok else '✗'} {ver}")

# Check ultralytics (YOLO)
ok, ver = check_import('ultralytics')
print(f"  ultralytics: {'✓' if ok else '✗'} {ver}")

# Check chess
ok, ver = check_import('chess')
print(f"  chess: {'✓' if ok else '✗'} {ver}")

print("-" * 50)

# Final CUDA test
import torch
if torch.cuda.is_available():
    print("\n✓ CUDA is working! Running quick tensor test...")
    x = torch.rand(100, 100, device='cuda')
    y = torch.rand(100, 100, device='cuda')
    z = torch.matmul(x, y)
    print(f"  Created {z.shape} tensor on GPU")
    print("\n✓ Jetson setup complete!")
else:
    print("\n✗ CUDA not available - check installation")
    sys.exit(1)
EOF

echo ""
echo "============================================================"
echo "Setup Complete!"
echo "============================================================"
echo ""
echo "You can now run the ChessBot demo:"
echo "  python scripts/best_move_demo.py --debug"
echo ""
