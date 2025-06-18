import json
import sys
import os
from pathlib import Path

# Add src to path to allow imports
sys.path.append(str(Path(__file__).parent))
from api.gemma3 import GemmaAnomalyDetector
from api.gemini import GeminiAPI
from utils.ocr import process_screenshots_folder, process_single_screenshot
from utils.cli import parse_arguments, debug_print

def main():
    args = parse_arguments()
    
    # Validate file paths based on mode
    if args.screenshots and not os.path.exists(args.screenshots):
        print(f"Error: Screenshots folder not found: {args.screenshots}")
        sys.exit(1)
    
    if args.single and not os.path.exists(args.single):
        print(f"Error: Screenshot file not found: {args.single}")
        sys.exit(1)
    
    if args.prohibited and not os.path.exists(args.prohibited):
        print(f"Error: Prohibited activities file not found: {args.prohibited}")
        sys.exit(1)
    
    # Initialize the detector based on model choice
    if args.model == 'gemini':
        debug_print("Using Google Gemini model for analysis...")
        detector = GeminiAPI()
    else:
        debug_print("Using default Gemma3 model for analysis...")
        detector = GemmaAnomalyDetector()
    
    try:
        # Handle single screenshot mode
        if args.single:
            debug_print(f"Processing single screenshot: {args.single}")
            screenshot = process_single_screenshot(args.single, args.pre_processor)
            
            # For single mode, we may not have a prohibited file
            if args.prohibited:
                # Full analysis with prohibited file
                results = detector.analyze_screenshot_ocr(screenshot, args.prohibited, args.role)
                standardized_result = detector.standardize_single_response(results, args.role)
            else:
                # Simple OCR extraction only
                standardized_result = {
                    "status": "success",
                    "screenshot": screenshot["filename"],
                    "ocr_text": screenshot["ocr_text"]
                }
                
        # Handle folder mode
        else:
            screenshots = process_screenshots_folder(args.screenshots, args.pre_processor)
            if not screenshots:
                print("No valid screenshots found in the specified folder")
                sys.exit(1)
                
            debug_print(f"Found {len(screenshots)} screenshots to analyze")
            
            # Analyze screenshots - plaintext OCR data
            debug_print(f"Starting analysis with {args.model} model...")
            results = detector.analyze_screenshots_with_ocr(
                screenshots,
                prohibited_path=args.prohibited,
                role=args.role
            )
            
            # For compatibility with existing code
            standardized_result = results
    
    except Exception as e:
        print(f"Error processing screenshot(s): {str(e)}")
        sys.exit(1)
    
    # Output results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(standardized_result, f, indent=2)
        debug_print(f"Results saved to {args.output}")
    else:
        print(json.dumps(standardized_result, indent=2))

if __name__ == "__main__":
    main()
