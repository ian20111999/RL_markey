#!/bin/bash
# Quick Start Script for RL Market Making Integrated Pipeline

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                                                              ║${NC}"
echo -e "${BLUE}║      🚀 RL Market Making - Integrated Training Pipeline      ║${NC}"
echo -e "${BLUE}║                     Quick Start Script                       ║${NC}"
echo -e "${BLUE}║                                                              ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Function to print section header
print_header() {
    echo -e "\n${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}\n"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check Python installation
print_header "Step 1: Checking Prerequisites"

if ! command_exists python3; then
    echo -e "${RED}❌ Python 3 not found. Please install Python 3.8+${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo -e "${GREEN}✅ Python $PYTHON_VERSION found${NC}"

# Check pip
if ! command_exists pip3; then
    echo -e "${RED}❌ pip3 not found. Please install pip${NC}"
    exit 1
fi
echo -e "${GREEN}✅ pip3 found${NC}"

# Install dependencies
print_header "Step 2: Installing Dependencies"

echo -e "${YELLOW}📦 Installing required packages...${NC}"
pip3 install -r requirements.txt --quiet

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ All dependencies installed successfully${NC}"
else
    echo -e "${RED}❌ Failed to install dependencies${NC}"
    exit 1
fi

# Create necessary directories
print_header "Step 3: Setting Up Directories"

mkdir -p data models logs runs logs/pipeline
echo -e "${GREEN}✅ Directories created${NC}"

# Check for data
print_header "Step 4: Checking Data"

DATA_FILES=$(ls -1 data/*.csv 2>/dev/null | wc -l)

if [ $DATA_FILES -eq 0 ]; then
    echo -e "${YELLOW}⚠️  No data files found in data/ directory${NC}"
    echo -e "${YELLOW}   You can:${NC}"
    echo -e "${YELLOW}   1. Download data: python scripts/fetch_data.py --symbol BTCUSDT --year 2023${NC}"
    echo -e "${YELLOW}   2. Place your own CSV files in the data/ directory${NC}"
    echo ""
    read -p "Do you want to download sample BTC data now? (y/N): " download_data
    
    if [ "$download_data" = "y" ] || [ "$download_data" = "Y" ]; then
        echo -e "${BLUE}📥 Downloading BTC data...${NC}"
        python3 scripts/fetch_data.py --symbol BTCUSDT --interval 1m --year 2023
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✅ Data downloaded successfully${NC}"
        else
            echo -e "${RED}❌ Failed to download data${NC}"
            exit 1
        fi
    fi
else
    echo -e "${GREEN}✅ Found $DATA_FILES data file(s)${NC}"
    ls data/*.csv | head -5
    if [ $DATA_FILES -gt 5 ]; then
        echo "   ... and $(($DATA_FILES - 5)) more"
    fi
fi

# Main menu
print_header "Step 5: Choose Training Mode"

echo -e "${GREEN}What would you like to do?${NC}"
echo ""
echo "  1) 🎯 Train a single symbol (e.g., BTC)"
echo "  2) 📊 Train multiple symbols"
echo "  3) 🔍 Auto-discover and train all available data"
echo "  4) 📈 Start monitoring dashboard only"
echo "  5) ❌ Exit"
echo ""
read -p "Enter your choice (1-5): " choice

case $choice in
    1)
        # Single symbol training
        print_header "Single Symbol Training"
        read -p "Enter symbol (e.g., btc, eth, bnb): " symbol
        
        if [ -z "$symbol" ]; then
            echo -e "${RED}❌ Symbol cannot be empty${NC}"
            exit 1
        fi
        
        read -p "Max retries (default: 5): " retries
        retries=${retries:-5}
        
        echo ""
        echo -e "${GREEN}🚀 Starting training for ${symbol^^}...${NC}"
        echo -e "${YELLOW}   This may take a while. You can monitor progress in real-time.${NC}"
        echo ""
        
        python3 integrated_pipeline.py --symbol "$symbol" --retries "$retries"
        ;;
        
    2)
        # Multiple symbols training
        print_header "Multiple Symbols Training"
        read -p "Enter symbols separated by space (e.g., btc eth bnb): " symbols
        
        if [ -z "$symbols" ]; then
            echo -e "${RED}❌ Symbols cannot be empty${NC}"
            exit 1
        fi
        
        echo ""
        echo -e "${GREEN}🚀 Starting training for: $symbols${NC}"
        echo -e "${YELLOW}   This will train each symbol sequentially.${NC}"
        echo ""
        
        python3 integrated_pipeline.py --symbols $symbols
        ;;
        
    3)
        # Auto-discover
        print_header "Auto-Discover Training"
        echo -e "${YELLOW}This will automatically find all CSV files in data/ and train them.${NC}"
        echo ""
        
        python3 integrated_pipeline.py --auto-discover
        ;;
        
    4)
        # Start dashboard
        print_header "Monitoring Dashboard"
        echo ""
        echo -e "${GREEN}Choose dashboard mode:${NC}"
        echo "  1) Console (text-based)"
        echo "  2) Web (browser-based)"
        echo ""
        read -p "Enter choice (1-2): " dashboard_choice
        
        case $dashboard_choice in
            1)
                python3 monitoring_dashboard.py
                ;;
            2)
                echo ""
                echo -e "${GREEN}🌐 Starting web dashboard server...${NC}"
                echo -e "${BLUE}   Access dashboard at: http://localhost:5000${NC}"
                echo -e "${BLUE}   Or open: dashboard_enhanced.html${NC}"
                echo ""
                echo -e "${YELLOW}   Press Ctrl+C to stop the server${NC}"
                echo ""
                python3 web_dashboard.py --host 0.0.0.0 --port 5000
                ;;
            *)
                echo -e "${RED}❌ Invalid choice${NC}"
                exit 1
                ;;
        esac
        ;;
        
    5)
        echo -e "${BLUE}👋 Goodbye!${NC}"
        exit 0
        ;;
        
    *)
        echo -e "${RED}❌ Invalid choice${NC}"
        exit 1
        ;;
esac

# Final message
if [ $choice -ne 4 ] && [ $choice -ne 5 ]; then
    print_header "Training Complete!"
    
    echo -e "${GREEN}✅ Training finished!${NC}"
    echo ""
    echo -e "${BLUE}📊 Next steps:${NC}"
    echo ""
    echo "  1. View results in terminal:"
    echo -e "     ${YELLOW}python3 monitoring_dashboard.py${NC}"
    echo ""
    echo "  2. Start web dashboard:"
    echo -e "     ${YELLOW}python3 web_dashboard.py${NC}"
    echo -e "     Then open: ${YELLOW}http://localhost:5000${NC}"
    echo ""
    echo "  3. Check trained models:"
    echo -e "     ${YELLOW}ls -lh models/${NC}"
    echo ""
    echo "  4. Deploy to production:"
    echo -e "     ${YELLOW}# Use models/*_best_model.zip for deployment${NC}"
    echo ""
fi
