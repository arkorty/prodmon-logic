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
from utils.learn import learn_unknown, get_relevant_words_from_ocr

def validate_paths(args):
    """Validate file and folder paths from arguments"""
    if args.screenshots and not os.path.exists(args.screenshots):
        print(f"Error: Screenshots folder not found: {args.screenshots}")
        sys.exit(1)
    
    if args.single and not os.path.exists(args.single):
        print(f"Error: Screenshot file not found: {args.single}")
        sys.exit(1)
    
    if args.prohibited and not os.path.exists(args.prohibited):
        print(f"Error: Prohibited activities file not found: {args.prohibited}")
        sys.exit(1)

def get_detector(model_choice):
    """Initialize the appropriate detector based on model choice"""
    if model_choice == 'gemini':
        debug_print("Using Google Gemini model for analysis...")
        return GeminiAPI()
    else:
        debug_print("Using default Gemma3 model for analysis...")
        return GemmaAnomalyDetector()

def process_single_mode(args, detector):
    """Handle single screenshot analysis mode"""
    debug_print(f"Processing single screenshot: {args.single}")
    screenshot = process_single_screenshot(args.single, args.pre_processor)
    # Full analysis with rules
    result = detector.analyze_screenshot_ocr(screenshot, role=args.role, company=getattr(args, 'company', None))
    return detector.standardize_single_response(result, args.role)

def process_folder_mode(args, detector):
    """Handle folder of screenshots analysis mode"""
    screenshots = process_screenshots_folder(args.screenshots, args.pre_processor)
    if not screenshots:
        print("No valid screenshots found in the specified folder")
        sys.exit(1)
    debug_print(f"Found {len(screenshots)} screenshots to analyze")
    debug_print(f"Starting analysis with {args.model} model...")
    results = []
    for screenshot in screenshots:
        result = detector.analyze_screenshot_ocr(screenshot, role=args.role, company=getattr(args, 'company', None))
        results.append(result)
    return {"results": results}

def save_output(result, output_path=None):
    """Save or print the analysis results"""
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        debug_print(f"Results saved to {output_path}")
    else:
        print(json.dumps(result, indent=2))

def main():
    """Main application entry point"""
    args = parse_arguments()
    validate_paths(args)
    detector = get_detector(args.model)
    
    try:
        # Process based on mode
        if args.single:
            result = process_single_screenshot(args.single, args.pre_processor)
        else:
            result = process_screenshots_folder(args.screenshots, args.pre_processor)
        
        # Output results
        save_output(result, args.output)

        debug_print(f"[LEARN] Result: {result}")

        # --- LEARNING SYSTEM: Check for unknowns in OCR text and update rules ---
        ocr_texts = []
        if args.single:
            # Try to get OCR text from all possible locations in the result
            if 'ocr_text' in result:
                ocr_texts.append(result['ocr_text'])
            elif 'screenshot' in result and 'ocr_text' in result['screenshot']:
                ocr_texts.append(result['screenshot']['ocr_text'])
            elif 'analysis' in result and 'ocr_text' in result['analysis']:
                ocr_texts.append(result['analysis']['ocr_text'])
        else:
            for res in result.get('results', []):
                if 'ocr_text' in res:
                    ocr_texts.append(res['ocr_text'])
                elif 'screenshot' in res and 'ocr_text' in res['screenshot']:
                    ocr_texts.append(res['screenshot']['ocr_text'])
                elif 'analysis' in res and 'ocr_text' in res['analysis']:
                    ocr_texts.append(res['analysis']['ocr_text'])
        role = args.role
        company = getattr(args, 'company', None)
        relevant_items = set()
        debug_print(f"[LEARN] OCR texts: {ocr_texts}")
        for ocr_text in ocr_texts:
            try:
                debug_print(f"[LEARN] Extracting relevant words from OCR text: {ocr_text[:100]}...")
                relevant = get_relevant_words_from_ocr(ocr_text, role=role, company=company)
                debug_print(f"[LEARN] Relevant words/entities found: {relevant}")
                relevant_items.update(relevant)
            except Exception as e:
                debug_print(f"[LEARN] Error extracting relevant words: {e}")
        import json
        from pathlib import Path
        rules_path = Path('rules') / role / 'baseline.json'
        known_items = set()
        if rules_path.exists():
            with open(rules_path, 'r') as f:
                rules_data = json.load(f)
                for rule in rules_data.get('rules', rules_data.get('allowed', [])):
                    known_items.add(rule.get('item') or rule.get('keyword'))
        debug_print(f"[LEARN] Known items for role {role}: {known_items}")
        unknowns = [item for item in relevant_items if item and item not in known_items]
        debug_print(f"[LEARN] Unknown items to learn: {unknowns}")
        for unknown in unknowns:
            debug_print(f"[LEARN] Unknown item detected: {unknown}. Querying LLM...")
            rule, updated_path = learn_unknown(unknown, role=role, company=company)
            debug_print(f"[LEARN] Rule added to {updated_path}: {rule}")
        # --- END LEARNING SYSTEM ---
        
    except Exception as e:
        print(f"Error processing screenshot(s): {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
