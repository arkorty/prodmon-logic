import json
import csv
import re
import os
import subprocess
from typing import Dict, List

class GemmaAnomalyDetector:
    def __init__(self, model_name: str = "gemma3:4b"):
        self.model_name = model_name
        
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
            print(f"Warning: Error reading CSV file: {e}")
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
    
    def create_prompt_for_ocr(self, prohibited_data: List[Dict], 
                             screenshot_data: Dict, role: str) -> str:
        """Create a prompt for the Gemma model to analyze OCR data"""
        # Format prohibited data for better readability
        prohibited_formatted = self.format_csv_for_prompt(prohibited_data)
        
        prompt = f"""
        You are an anomaly detection system analyzing user behavior from screenshots.
        
        PROHIBITED ACTIVITIES (examples of anomalous behavior):
        {prohibited_formatted}
        
        JOB ROLE: {role}
        
        SCREENSHOT DATA:
        Filename: {screenshot_data['filename']}
        Text extracted from screenshot:
        "{screenshot_data['ocr_text']}"
        
        Task: Analyze the screenshot text and identify any anomalies by checking for prohibited activities.
        Consider the context of a {role} role - absence of prohibited activities is considered normal baseline behavior.
        
        For each detected anomaly, provide:
        1. The type of anomaly (Unusual Task, Unusual Interaction, Irregular Activity, Unauthorized Access, or Other)
        2. A detailed explanation of why it's considered anomalous
        3. The severity level (Low, Medium, High)
        
        Format your response as a valid JSON object with the following structure:
        {{
            "anomalies": [
                {{
                    "type": "anomaly type",
                    "explanation": "detailed explanation",
                    "severity": "severity level",
                    "evidence": "specific text that indicates the anomaly"
                }}
            ],
            "is_anomalous": true/false,
            "confidence_score": 0-100,
            "role_specific_insights": "insights specific to the {role} role"
        }}
        """

        return prompt
    
    def call_gemma(self, prompt: str) -> str:
        """Call the Gemma model with the prompt"""
        try:
            result = subprocess.run(
                ["ollama", "run", self.model_name],
                input=prompt,
                capture_output=True,
                text=True,
                timeout=60  # Set timeout to 60 seconds
            )
            return result.stdout
        except Exception as e:
            return f'{{"error": "Error calling Gemma model: {str(e)}"}}'
    
    def extract_json_from_response(self, response: str) -> Dict:
        """Extract and parse JSON from the model response"""
        # Find JSON content using regex
        json_match = re.search(r'({[\s\S]*})', response)
        if json_match:
            try:
                json_str = json_match.group(1)
                return json.loads(json_str)
            except json.JSONDecodeError:
                return {"error": "Failed to parse JSON from model response"}
        return {"error": "No JSON found in model response"}
    
    def analyze_screenshot_ocr(self, screenshot_data: Dict, 
                              prohibited_path: str, role: str) -> Dict:
        """Analyze a screenshot using OCR data and detect anomalies"""
        # Load prohibited activities
        prohibited_data = self.load_csv_data(prohibited_path)
        
        # Create prompt for the model
        prompt = self.create_prompt_for_ocr(prohibited_data, screenshot_data, role)
        
        # Call the model and get response
        response = self.call_gemma(prompt)
        
        # Extract and return JSON results
        result = self.extract_json_from_response(response)
        
        # Add screenshot metadata to result
        result["screenshot_filename"] = screenshot_data["filename"]
        result["screenshot_path"] = screenshot_data.get("file_path", "")
        
        return result
    
    def analyze_screenshots_with_ocr(self, screenshots: List[Dict], 
                                   prohibited_path: str, role: str) -> Dict:
        """Analyze multiple screenshots using OCR data and provide a comprehensive report"""
        results = []
        overall_anomalous = False
        
        for screenshot in screenshots:
            result = self.analyze_screenshot_ocr(screenshot, prohibited_path, role)
            results.append(result)
            if result.get("is_anomalous", False):
                overall_anomalous = True
        
        return {
            "results": results,
            "overall_anomalous": overall_anomalous,
            "screenshot_count": len(screenshots),
            "anomalous_screenshot_count": sum(1 for r in results if r.get("is_anomalous", False)),
            "role_analyzed": role
        }
