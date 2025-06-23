import os
import json
from time import sleep
import google.generativeai as genai
from typing import Dict, Optional, List
import csv
import re
from datetime import datetime
from utils.cli import debug_print
from dotenv import load_dotenv
from pathlib import Path
from utils.learn import load_merged_rules


class GeminiAPI:
    """Class to handle interactions with the Google Gemini API"""
    
    def __init__(self, model_name: str = "gemini-2.0-flash-lite"):
        self.model_name = model_name
        # Load environment variables from .env file
        env_path = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))) / '.env'
        load_dotenv(dotenv_path=env_path)
        
        # Initialize the Gemini API with API key
        api_key = os.environ.get('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in .env file or environment variables")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
    
    def generate_content(self, prompt: str) -> str:
        """Call the Gemini API with a prompt and return the response text"""
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            error_message = str(e)
            debug_print(f"Error calling Gemini API: {error_message}")
            return f'{{"error": "Error calling Gemini API: {error_message}"}}'
    
    def analyze_text(self, prompt: str) -> Dict:
        """Call the Gemini API and parse the response as JSON"""
        response_text = self.generate_content(prompt)
        sleep(1000)  # Sleep to avoid rate limiting issues
        return self._extract_json_from_response(response_text)
    
    def _extract_json_from_response(self, response: str) -> Dict:
        """Extract and parse JSON from the model response"""
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # Try to extract JSON using regex if direct parsing fails
            json_match = re.search(r'({[\s\S]*})', response)
            if json_match:
                try:
                    json_str = json_match.group(1)
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    pass
            return {"error": "Failed to parse JSON from model response", 
                    "raw_response": response}
    
    def load_csv_data(self, file_path: str) -> List[Dict]:
        """Load and parse CSV data into a list of dictionaries"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CSV file not found: {file_path}")
        
        data = []
        try:
            with open(file_path, 'r') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    data.append(dict(row))
        except Exception as e:
            debug_print(f"Warning: Error reading CSV file: {e}")
            # Return a minimal structure so processing can continue
            data = [{"warning": f"CSV processing error: {e}"}]
            
        return data
    
    def format_csv_for_prompt(self, csv_data: List[Dict]) -> str:
        """Format CSV data in a more readable way for the prompt"""
        if not csv_data:
            return "No data available"
            
        # Extract a simple text representation of the CSV data
        result = []
        for i, row in enumerate(csv_data):
            entry = f"Entry {i+1}:\n"
            for key, value in row.items():
                entry += f"  - {key}: {value}\n"
            result.append(entry)
            
        return "\n".join(result)
    
    def create_prompt_for_ocr(self, rules: dict, screenshot_data: Dict, role: str) -> str:
        """Create a prompt for the Gemini model to analyze OCR data"""
        prohibited_formatted = '\n'.join([f"- {item.get('item', item.get('keyword', ''))}" for item in rules.get('prohibited', [])])
        allowed_formatted = '\n'.join([f"- {item.get('item', item.get('keyword', ''))}" for item in rules.get('allowed', [])])
        prompt = f"""
        You are an anomaly detection system analyzing desktop screenshots for productivity monitoring.

        JOB ROLE: {role}
        PROHIBITED: {prohibited_formatted}
        ALLOWED: {allowed_formatted}
        BASELINE EXPECTATIONS: Normal behavior includes role-specific applications and workflows

        SCREENSHOT DATA:
        • Filename: {screenshot_data['filename']}
        • Extracted text: \"{screenshot_data['ocr_text']}\"

        ANALYSIS TASK:
        1. Detect deviations from {role} baseline behavior
        2. Focus on these anomaly types:
           - Unusual Task Activity
           - Unusual Interactions
           - Irregular Pauses
           - Atypical Interaction Patterns
           - General Behavioral Outliers
        3. For each detected anomaly:
           - Specify the anomaly type
           - Provide concise explanation (include evidence)
           - Assign confidence (0.0-1.0)

        OUTPUT REQUIREMENTS:
        Return valid JSON with this structure:
        {{
            \"anomaly_detected\": true/false,
            \"baseline_role\": \"{role}\",
            \"anomalies\": [
                {{
                    \"type\": \"anomaly category\",
                    \"explanation\": \"clear, evidence-based reason\",
                    \"confidence\": 0.0-1.0
                }}
            ]
        }}
        """
        return prompt
    
    def analyze_screenshot_ocr(self, screenshot_data: Dict, role: str, company: str = None) -> Dict:
        """Analyze a screenshot using OCR data and detect anomalies (JSON rules only)"""
        rules = load_merged_rules(role, company)
        prompt = self.create_prompt_for_ocr(rules, screenshot_data, role)
        response = self.generate_content(prompt)
        result = self._extract_json_from_response(response)
        result["screenshot_filename"] = screenshot_data["filename"]
        return result
    
    def standardize_single_response(self, response: Dict, role: str) -> Dict:
        """Standardize a single screenshot analysis response"""
        # Handle error case
        if "error" in response:
            return {"status": "error", "message": response["error"]}
        
        # Get timestamp from filename or use current time
        timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
        filename = response.get("screenshot_filename", "")
        timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})', filename)
        if timestamp_match:
            timestamp = f"{timestamp_match.group(1)}Z"
        
        # Determine if the response has anomalies
        anomaly_detected = response.get("anomaly_detected", response.get("is_anomalous", False))
        baseline_role = response.get("baseline_role", role)
        
        # Process anomalies
        formatted_anomalies = []
        anomalies = response.get("anomalies", [])
        
        for anomaly in anomalies:
            # Check if the anomaly already has a confidence field
            if "confidence" in anomaly:
                confidence = anomaly.get("confidence")
            else:
                # Map severity to confidence value
                severity = anomaly.get("severity", "Medium").lower()
                confidence_map = {"high": 0.95, "medium": 0.75, "low": 0.55}
                confidence = confidence_map.get(severity, 0.5)
                
                # Use confidence score from the response if available
                if "confidence_score" in response:
                    confidence = min(max(response.get("confidence_score", 50) / 100, 0), 1)
            
            formatted_anomalies.append({
                "type": anomaly.get("type", "Unknown"),
                "explanation": anomaly.get("explanation", "No explanation provided"),
                "confidence": round(float(confidence), 2),
                "timestamp": timestamp
            })
        
        return {
            "status": "success",
            "analysis": {
                "anomaly_detected": anomaly_detected,
                "baseline_role": baseline_role,
                "anomalies": formatted_anomalies
            }
        }
    
    def _add_timestamps_to_anomalies(self, anomalies: List[Dict]) -> List[Dict]:
        """Add timestamps to anomalies if missing"""
        timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
        
        updated_anomalies = []
        for anomaly in anomalies:
            if "timestamp" not in anomaly:
                anomaly_copy = anomaly.copy()
                anomaly_copy["timestamp"] = timestamp
                updated_anomalies.append(anomaly_copy)
            else:
                updated_anomalies.append(anomaly)
                
        return updated_anomalies
    
    def format_standardized_output(self, analysis_results: Dict) -> Dict:
        """Convert the analysis results into a standardized output format"""
        # Handle error case
        if "error" in analysis_results:
            return {"status": "error", "message": analysis_results["error"]}
        
        # Handle direct model responses vs batched results
        if "results" in analysis_results:
            formatted_anomalies = []
            
            # Process each screenshot result
            for result in analysis_results.get("results", []):
                if not result.get("is_anomalous", False) and not result.get("anomaly_detected", False):
                    continue
                    
                # Get timestamp from filename or use current time
                timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
                filename = result.get("screenshot_filename", "")
                timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})', filename)
                if timestamp_match:
                    timestamp = f"{timestamp_match.group(1)}Z"
                
                # Process anomalies if available
                if "anomalies" in result:
                    for anomaly in result.get("anomalies", []):
                        # Handle confidence calculation
                        if "confidence" in anomaly:
                            confidence = anomaly.get("confidence")
                        else:
                            severity = anomaly.get("severity", "Medium").lower()
                            confidence_map = {"high": 0.95, "medium": 0.75, "low": 0.55}
                            confidence = confidence_map.get(severity, 0.5)
                            
                            if "confidence_score" in result:
                                confidence = min(max(result.get("confidence_score", 50) / 100, 0), 1)
                        
                        formatted_anomalies.append({
                            "type": anomaly.get("type", "Unknown"),
                            "explanation": anomaly.get("explanation", "No explanation provided"),
                            "confidence": round(float(confidence), 2),
                            "timestamp": timestamp
                        })
            
            return {
                "status": "success",
                "analysis": {
                    "anomaly_detected": analysis_results.get("overall_anomalous", False),
                    "baseline_role": analysis_results.get("role_analyzed", "unknown"),
                    "anomalies": formatted_anomalies
                }
            }
        else:
            # Direct model response with newer format
            return {
                "status": "success",
                "analysis": {
                    "anomaly_detected": analysis_results.get("anomaly_detected", False),
                    "baseline_role": analysis_results.get("baseline_role", "unknown"),
                    "anomalies": self._add_timestamps_to_anomalies(analysis_results.get("anomalies", []))
                }
            }
    
    def parse_direct_responses(self, responses: List[str], role: str = "developer") -> Dict:
        """Parse a list of direct Gemini responses and format them into the standard output"""
        results = []
        overall_anomalous = False
        
        for response in responses:
            # Try to extract JSON from the response
            try:
                # First try with code block markers
                json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response)
                if json_match:
                    result = json.loads(json_match.group(1))
                else:
                    # Try direct parsing
                    result = json.loads(response)
                
                if result.get("anomaly_detected", False):
                    overall_anomalous = True
                results.append(result)
            except (json.JSONDecodeError, TypeError) as e:
                results.append({
                    "error": f"Failed to parse JSON from response: {str(e)}",
                    "raw_response": response
                })
        
        analysis_results = {
            "results": results,
            "overall_anomalous": overall_anomalous,
            "screenshot_count": len(responses),
            "anomalous_screenshot_count": sum(1 for r in results if r.get("anomaly_detected", False)),
            "role_analyzed": role
        }
        
        # Convert to standardized format
        return self.format_standardized_output(analysis_results)
    
    def analyze_screenshots_with_ocr(self, screenshots: List[Dict], 
                                   prohibited_path: str, role: str) -> Dict:
        """Analyze multiple screenshots using OCR data and provide a comprehensive report"""
        results = []
        overall_anomalous = False
        
        for screenshot in screenshots:
            result = self.analyze_screenshot_ocr(screenshot, prohibited_path, role)
            results.append(result)
            if result.get("is_anomalous", False) or result.get("anomaly_detected", False):
                overall_anomalous = True
        
        return {
            "results": results,
            "overall_anomalous": overall_anomalous,
            "screenshot_count": len(screenshots),
            "anomalous_screenshot_count": sum(1 for r in results if r.get("is_anomalous", False) or r.get("anomaly_detected", False)),
            "role_analyzed": role
        }
    
    def analyze_screenshots_with_ocr(self, screenshots: List[Dict], 
                                   prohibited_path: str, role: str) -> Dict:
        """Analyze multiple screenshots using OCR data and provide a comprehensive report"""
        results = []
        overall_anomalous = False
        
        for screenshot in screenshots:
            result = self.analyze_screenshot_ocr(screenshot, prohibited_path, role)
            results.append(result)
            if result.get("is_anomalous", False) or result.get("anomaly_detected", False):
                overall_anomalous = True
        
        # Return the original format
        return {
            "results": results,
            "overall_anomalous": overall_anomalous,
            "screenshot_count": len(screenshots),
            "anomalous_screenshot_count": sum(1 for r in results if r.get("is_anomalous", False) or r.get("anomaly_detected", False)),
            "role_analyzed": role
        }
