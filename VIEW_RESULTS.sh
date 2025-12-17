#!/bin/bash
# Quick script to view all experimental results

echo "=================================================="
echo "Prometheus-Eval - Experimental Results Viewer"
echo "=================================================="
echo ""

# Check if results exist
if [ ! -d "results" ]; then
    echo "❌ No results directory found!"
    echo "Run experiments first: python run_experiments.py --dataset all"
    exit 1
fi

# Function to display menu
show_menu() {
    echo "What would you like to view?"
    echo ""
    echo "  1) Summary Visualizations (3 charts)"
    echo "  2) Sentiment Analysis Results"
    echo "  3) Math Reasoning Results (BEST GAINS!)"
    echo "  4) Logic Reasoning Results"
    echo "  5) All Visualizations (9 charts)"
    echo "  6) Experimental Data (JSON files)"
    echo "  7) Exit"
    echo ""
}

# Main loop
while true; do
    show_menu
    read -p "Enter your choice (1-7): " choice
    echo ""

    case $choice in
        1)
            echo "📊 Opening Summary Visualizations..."
            open results/visualizations/*.png 2>/dev/null || echo "❌ No summary visualizations found"
            ;;
        2)
            echo "📊 Opening Sentiment Analysis Results..."
            open results/experiments/sentiment_*/*.png 2>/dev/null || echo "❌ No sentiment results found"
            ;;
        3)
            echo "📊 Opening Math Reasoning Results (BEST GAINS!)..."
            open results/experiments/math_*/*.png 2>/dev/null || echo "❌ No math results found"
            ;;
        4)
            echo "📊 Opening Logic Reasoning Results..."
            open results/experiments/logic_*/*.png 2>/dev/null || echo "❌ No logic results found"
            ;;
        5)
            echo "📊 Opening ALL Visualizations..."
            find results -name "*.png" -exec open {} \;
            ;;
        6)
            echo "📄 Listing Experimental Data Files..."
            echo ""
            find results/experiments -name "*.json" -type f | while read file; do
                echo "  📁 $file"
                echo "     $(wc -l < "$file") lines"
            done
            echo ""
            ;;
        7)
            echo "👋 Goodbye!"
            exit 0
            ;;
        *)
            echo "❌ Invalid choice. Please enter 1-7."
            ;;
    esac
    echo ""
    echo "Press Enter to continue..."
    read
    clear
done
