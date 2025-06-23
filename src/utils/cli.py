import argparse
import json
import sys
import csv
from typing import Dict, List
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

# Debug flag to control output verbosity
DEBUG_MODE = False

def debug_print(*args, **kwargs):
    """Print only when in debug mode"""
    if DEBUG_MODE:
        print(*args, **kwargs)

def parse_arguments():
    """Parse command line arguments for the application"""
    parser = argparse.ArgumentParser(description="Analyze user behaviors for anomalies using Gemma3 or Gemini")
    parser.add_argument('--prohibited', required=False, help='Path to the prohibited activities CSV')
    parser.add_argument('--screenshots', required=False, help='Path to folder containing screenshots')
    parser.add_argument('--single', help='Path to a single screenshot to analyze')
    parser.add_argument('--output', help='Path to save the output JSON (default: stdout)')
    parser.add_argument('--role', default='developer', help='Job role for baseline comparison')
    parser.add_argument('--pre-processor', default='thresh', choices=['thresh', 'blur'], 
                      help='OCR pre-processing method (thresh or blur)')
    parser.add_argument('--model', default='gemma', choices=['gemma', 'gemini'], 
                      help='AI model to use (gemma or gemini)')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    
    args = parser.parse_args()
    
    # Set the global debug flag
    global DEBUG_MODE
    DEBUG_MODE = args.debug
    
    # Validate that either --screenshots or --single is provided
    if not args.screenshots and not args.single:
        parser.error("Either --screenshots or --single argument must be provided")
        
    # Validate that --prohibited is provided unless in single mode
    if not args.prohibited and not args.single:
        parser.error("--prohibited argument is required when not using --single mode")
        
    return args

def load_input_data(input_path: str) -> List[Dict]:
    """Load input data from either JSON or CSV file"""
    if input_path.endswith('.json'):
        with open(input_path, 'r') as f:
            return json.load(f)
    elif input_path.endswith('.csv'):
        data = []
        with open(input_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(dict(row))
        return data
    else:
        raise ValueError("Input file must be either JSON or CSV")
