import os
import json
from pathlib import Path
from utils.cli import debug_print

def generate_rule_prompt(item, role=None, company=None):
    context = f'role: "{role}"' if role else 'no specific role'
    if company:
        context += f', company: "{company}"'
    prompt = f'''
A screenshot contains the item: "{item}".
This item is not present in the current rules for the {context}.

Please classify this item as either "allowed" or "prohibited" for this context, and return your answer in the following JSON format:

{{
  "type": "allowed" | "prohibited",
  "category": "string",
  "subcategory": "string",
  "item": "{item}",
  "severity": "High" | "Medium" | "Low" | "Critical",
  "score": 0-100,
  "rationale": "string",
  "examples": ["Example 1", "Example 2"]
}}

Only return valid JSON. Do not include any explanation outside the JSON.
'''
    return prompt

def get_rule_from_llm(item, role=None, company=None):
    from api.gemini import GeminiAPI  # moved import here to avoid circular import
    import re  # Add import for regex
    
    prompt = generate_rule_prompt(item, role, company)
    debug_print(f"[LEARN] Sending rule prompt to LLM for item '{item}': {prompt}")
    llm = GeminiAPI()  # Or your LLM interface
    response = llm.generate_content(prompt)
    debug_print(f"[LEARN] LLM response for rule: {response}")
    
    # Try to parse JSON from response
    try:
        # Clean the response to handle potential code blocks or extra text
        cleaned_response = response.strip()
        
        # Remove markdown code block formatting if present
        code_block_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
        code_block_match = re.search(code_block_pattern, cleaned_response)
        if code_block_match:
            cleaned_response = code_block_match.group(1).strip()
            debug_print(f"[LEARN] Extracted JSON from code block: {cleaned_response}")
        
        # Parse the JSON
        rule = json.loads(cleaned_response)
        debug_print(f"[LEARN] Parsed rule from LLM: {rule}")
        return rule
    except Exception as e:
        debug_print(f"[LEARN] Failed to parse LLM rule JSON: {response}")
        debug_print(f"[LEARN] Error: {str(e)}")
        raise ValueError(f"LLM did not return valid JSON: {response}")

def update_rules(rule, role=None, company=None):
    # Determine rules file path
    base_dir = Path(__file__).parent.parent.parent / 'rules'
    if role and company:
        rules_path = base_dir / role / f'{company}.json'
    elif role:
        rules_path = base_dir / role / 'baseline.json'
    else:
        rules_path = base_dir / 'baseline.json'
    debug_print(f"[LEARN] Updating rules file: {rules_path} with rule: {rule}")
    # Load or create rules file
    if rules_path.exists():
        with open(rules_path, 'r') as f:
            rules_data = json.load(f)
    else:
        rules_data = {"context": role or "baseline", "description": "", "rules": []}
    # Add rule
    rules_data.setdefault("rules", []).append(rule)
    # Save back
    with open(rules_path, 'w') as f:
        json.dump(rules_data, f, indent=2)
    debug_print(f"[LEARN] Rule successfully added to {rules_path}")
    return rules_path

def learn_unknown(item, role=None, company=None):
    """
    Main entry: Given an unknown item, ask LLM, update rules, and return the new rule.
    """
    rule = get_rule_from_llm(item, role, company)
    rules_path = update_rules(rule, role, company)
    return rule, rules_path

def get_relevant_words_from_ocr(ocr_text, role=None, company=None):
    from api.gemini import GeminiAPI  # moved import here to avoid circular import
    import re  # Add import for regex
    
    context = f'role: "{role}"' if role else 'no specific role'
    if company:
        context += f', company: "{company}"'
    prompt = f'''
Given the following OCR text from a screenshot for {context}:
"""
{ocr_text}
"""

Return a JSON array of only the relevant words, phrases, or entities that should be checked against productivity rules (e.g., app names, website names, software, platforms, etc). Ignore common words, stopwords, and irrelevant text. Only return the JSON array, nothing else.
'''
    debug_print(f"[LEARN] Sending relevant words prompt to LLM: {prompt[:200]}...")
    llm = GeminiAPI()  # Or your LLM interface
    response = llm.generate_content(prompt)
    debug_print(f"[LEARN] LLM response for relevant words: {response}")
    
    try:
        # Clean the response to handle potential code blocks or extra text
        cleaned_response = response.strip()
        
        # Remove markdown code block formatting if present
        code_block_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
        code_block_match = re.search(code_block_pattern, cleaned_response)
        if code_block_match:
            cleaned_response = code_block_match.group(1).strip()
            debug_print(f"[LEARN] Extracted JSON from code block: {cleaned_response}")
        
        # Find JSON array pattern as fallback
        json_array_pattern = r'\[(.*)\]'
        json_array_match = re.search(json_array_pattern, cleaned_response)
        if not cleaned_response.startswith('[') and json_array_match:
            cleaned_response = f"[{json_array_match.group(1)}]"
            debug_print(f"[LEARN] Extracted JSON array: {cleaned_response}")
        
        # Parse the JSON
        relevant_words = json.loads(cleaned_response)
        debug_print(f"[LEARN] Parsed relevant words: {relevant_words}")
        
        if isinstance(relevant_words, list):
            return relevant_words
        else:
            debug_print(f"[LEARN] Expected list but got: {type(relevant_words)}")
            return []
    except Exception as e:
        debug_print(f"[LEARN] Failed to parse relevant words JSON: {response}")
        debug_print(f"[LEARN] Error: {str(e)}")
        raise ValueError(f"LLM did not return valid JSON array: {response}")

def load_merged_rules(role=None, company=None):
    base_dir = Path(__file__).parent.parent.parent / 'rules'
    merged = {"allowed": [], "prohibited": []}

    # Load baseline
    baseline_path = base_dir / 'baseline.json'
    if baseline_path.exists():
        with open(baseline_path) as f:
            data = json.load(f)
            merged["allowed"].extend(data.get("allowed", []))
            merged["prohibited"].extend(data.get("prohibited", []))

    # Load role baseline
    if role:
        role_baseline = base_dir / role / 'baseline.json'
        if role_baseline.exists():
            with open(role_baseline) as f:
                data = json.load(f)
                merged["allowed"].extend(data.get("allowed", []))
                merged["prohibited"].extend(data.get("prohibited", []))

    # Load company-specific
    if role and company:
        company_path = base_dir / role / f"{company}.json"
        if company_path.exists():
            with open(company_path) as f:
                data = json.load(f)
                merged["allowed"].extend(data.get("allowed", []))
                merged["prohibited"].extend(data.get("prohibited", []))

    return merged
