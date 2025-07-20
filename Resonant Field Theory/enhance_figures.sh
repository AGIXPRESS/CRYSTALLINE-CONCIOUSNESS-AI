#!/bin/bash
# Script to enhance visualizations for Resonant Field Theory paper
# This script sets up dependencies and runs the visualization enhancement script

# Exit on any error
set -e

# Define colors for output messages
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Print with timestamp
log() {
    echo -e "$(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# Print success message
success() {
    echo -e "${GREEN}$(date '+%Y-%m-%d %H:%M:%S') - ✓ $1${NC}"
}

# Print error message
error() {
    echo -e "${RED}$(date '+%Y-%m-%d %H:%M:%S') - ✗ $1${NC}"
}

# Print warning message
warning() {
    echo -e "${YELLOW}$(date '+%Y-%m-%d %H:%M:%S') - ⚠ $1${NC}"
}

# Print info message
info() {
    echo -e "${BLUE}$(date '+%Y-%m-%d %H:%M:%S') - ℹ $1${NC}"
}

# Print header
header() {
    echo -e "\n${CYAN}=============================================================${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}=============================================================${NC}"
}

# Function to clean up on error
cleanup() {
    echo ""
    error "An error occurred during the setup process."
    
    if [ -n "$VIRTUAL_ENV" ]; then
        info "Deactivating virtual environment..."
        deactivate 2>/dev/null || true
    fi
    
    exit 1
}

# Set trap for error handling
trap cleanup ERR

# Function to check command exists
command_exists() {
    command -v "$1" &> /dev/null
}

# Function to check Python version
check_python_version() {
    local required_major=3
    local required_minor=9
    
    if command_exists python3; then
        local version=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
        local major=$(echo $version | cut -d. -f1)
        local minor=$(echo $version | cut -d. -f2)
        
        if [ "$major" -lt "$required_major" ] || ([ "$major" -eq "$required_major" ] && [ "$minor" -lt "$required_minor" ]); then
            error "Python version $version is below the required version $required_major.$required_minor+"
            return 1
        else
            success "Found Python $version (meets requirement of $required_major.$required_minor+)"
            return 0
        fi
    else
        error "Python 3 not found"
        return 1
    fi
}

# Function to create requirements file if it doesn't exist
create_requirements_file() {
    if [ ! -f requirements.txt ]; then
        info "Creating requirements.txt file..."
        cat > requirements.txt << EOF
numpy>=2.0.0
matplotlib>=3.7.0
mlx>=0.25.0
pdf2image>=1.16.3
Pillow>=10.0.0
opencv-python>=4.8.0
scipy>=1.11.0
EOF
        success "requirements.txt file created."
    fi
}

# Function to check for Apple Silicon
is_apple_silicon() {
    if [[ $(uname -m) == "arm64" ]]; then
        return 0
    else
        return 1
    fi
}

# Main script starts here
header "Resonant Field Theory Visualization Enhancement"

info "Starting visualization enhancement setup (macOS optimized)"

# Check for Homebrew (macOS package manager)
if ! command_exists brew; then
    warning "Homebrew not found. Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    if [ $? -ne 0 ]; then
        error "Failed to install Homebrew. Please install it manually from https://brew.sh"
        exit 1
    else
        success "Homebrew installed successfully."
        
        # Add Homebrew to PATH if needed (especially on Apple Silicon)
        if is_apple_silicon; then
            if [ -f ~/.zshrc ]; then
                if ! grep -q '/opt/homebrew/bin' ~/.zshrc; then
                    echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zshrc
                    eval "$(/opt/homebrew/bin/brew shellenv)"
                    success "Homebrew added to PATH."
                fi
            fi
        fi
    fi
else
    success "Homebrew is already installed."
fi

# Check if poppler is installed (required for pdf2image)
if ! brew list --versions poppler &> /dev/null; then
    warning "Poppler not found. Installing poppler..."
    brew install poppler
    if [ $? -ne 0 ]; then
        error "Failed to install poppler. Please install it manually with 'brew install poppler'."
        exit 1
    else
        success "Poppler installed successfully."
    fi
else
    success "Poppler is already installed."
fi

# Check for Python with required version
if ! check_python_version; then
    error "Required Python version not found. Please install Python 3.9 or higher."
    info "You can install it with: brew install python@3.9"
    exit 1
fi

# Create a virtual environment
VENV_DIR="venv"
if [ ! -d "$VENV_DIR" ]; then
    info "Creating Python virtual environment in $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
    if [ $? -ne 0 ]; then
        warning "Failed to create virtual environment. Will continue with system Python."
    else
        success "Virtual environment created."
    fi
fi

# Activate virtual environment if it exists
if [ -d "$VENV_DIR" ]; then
    info "Activating virtual environment..."
    source "$VENV_DIR/bin/activate"
    if [ $? -ne 0 ]; then
        warning "Failed to activate virtual environment. Will continue with system Python."
    else
        success "Virtual environment activated."
    fi
fi

# Create requirements file if it doesn't exist
create_requirements_file

# Upgrade pip
info "Upgrading pip..."
python3 -m pip install --upgrade pip
if [ $? -ne 0 ]; then
    warning "Failed to upgrade pip. Continuing with existing version."
else
    success "Pip upgraded successfully."
fi

# Install required packages
header "Installing Python dependencies"

info "Installing required Python packages..."
python3 -m pip install -r requirements.txt
if [ $? -ne 0 ]; then
    error "Failed to install required Python packages. Please check the requirements.txt file and try again."
    exit 1
else
    success "Required Python packages installed successfully."
fi

# Check for MLX/MPS support
info "Checking for GPU acceleration support..."
if is_apple_silicon; then
    info "Apple Silicon detected - checking for MLX support..."
    if python3 -c "import mlx.core; print('MLX available')" &> /dev/null; then
        success "MLX is available for GPU acceleration."
    else
        warning "MLX is not properly installed. Visualization will use CPU fallback."
        warning "For GPU acceleration, ensure mlx is installed: pip install mlx"
    fi
else
    warning "Running on Intel Mac - MLX GPU acceleration may not be available."
    warning "For best performance, consider running on Apple Silicon (M1/M2/M3) Mac."
fi

# Check if figures directory exists and contains the necessary files
FIGURES_DIR="figures"
if [ ! -d "$FIGURES_DIR" ]; then
    warning "Figures directory not found. Creating directory structure..."
    mkdir -p "$FIGURES_DIR"
    info "Please place the following PDF files in the $FIGURES_DIR directory:"
    info "  - geometric_basis.pdf"
    info "  - field_evolution_0.pdf"
    info "  - field_evolution_8.pdf"
    info "  - field_evolution_17.pdf"
    exit 1
fi

REQUIRED_FILES=(
    "geometric_basis.pdf"
    "field_evolution_0.pdf"
    "field_evolution_8.pdf"
    "field_evolution_17.pdf"
)

MISSING_FILES=()
for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$FIGURES_DIR/$file" ]; then
        MISSING_FILES+=("$file")
    fi
done

if [ ${#MISSING_FILES[@]} -gt 0 ]; then
    error "The following required files are missing in the figures directory:"
    for file in "${MISSING_FILES[@]}"; do
        echo "  - $file"
    done
    error "Please place the required PDF files in the $FIGURES_DIR directory before running this script."
    exit 1
else
    success "All required figure files found."
fi

# Check if the enhancement script exists
if [ ! -f "enhance_visualizations.py" ]; then
    error "enhance_visualizations.py not found. Please make sure the script exists."
    exit 1
else
    # Make the script executable
    chmod +x enhance_visualizations.py
    success "Found enhancement script."
fi

# Run the enhancement script
header "Running Visualization Enhancement"

info "Starting the enhancement process. This may take several minutes..."
info "The script will use GPU acceleration if available."

python3 enhance_visualizations.py
if [ $? -ne 0 ]; then
    error "Visualization enhancement script failed. Please check the error messages above."
    exit 1
else
    success "Visualization enhancement completed successfully!"
fi

# Find the most recent output directory
LATEST_OUTPUT=$(ls -td figures_enhanced_* | head -n1)

if [ -n "$LATEST_OUTPUT" ]; then
    success "Enhanced figures have been saved to: $LATEST_OUTPUT"
    info "A LaTeX update guide has been generated in this directory."
    
    # Count the number of enhanced files
    NUM_FILES=$(find "$LATEST_OUTPUT" -name "*.pdf" | wc -l)
    info "Generated $NUM_FILES enhanced visualization files."
    
    # Open the directory for the user
    info "Opening the output directory..."
    open "$LATEST_OUTPUT" 2>/dev/null || true
else
    warning "Could not find the output directory. Please check the script output."
fi

# If we activated a virtual environment, deactivate it
if [ -n "$VIRTUAL_ENV" ]; then
    info "Deactivating virtual environment..."
    deactivate
    success "Virtual environment deactivated."
fi

header "Visualization Enhancement Process Complete"

info "To update your LaTeX document with the enhanced figures, follow the instructions in the"
info "latex_update_guide.txt file in the output directory."
info ""
info "The enhanced figures provide clear visual distinctions to highlight the subtle"
info "differences between stages, with color-coding, geometric guides, and magnified"
info "regions of interest."

# Remove the trap before exiting normally
trap - ERR

exit 0

