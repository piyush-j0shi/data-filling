"""
Enhanced Medical Document Extraction Service (OpenAI Version)
===========================================================

Optimized for accurate extraction of handwritten marks, circles, and connections
from medical fee tickets and patient questionnaires.

"""

import json
import os
import base64
import logging
import io
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

from openai import OpenAI
from pdf2image import convert_from_path
from PIL import Image
import numpy as np


class ExtractionStatus(Enum):
    """Status enumeration for extraction operations."""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"


@dataclass
class PatientDemographics:
    """Structured patient demographic information."""
    patient_name: str = "N/A"
    date_of_birth: str = "N/A"
    age: str = "N/A"
    mrn: str = "N/A"
    ssn: str = "N/A"
    gender: str = "N/A"


@dataclass
class ContactInformation:
    """Structured contact information."""
    address: str = "N/A"
    city: str = "N/A"
    state: str = "N/A"
    zip_code: str = "N/A"
    phone: str = "N/A"
    email: str = "N/A"


@dataclass
class InsuranceInformation:
    """Structured insurance information."""
    primary_insurance: str = "N/A"
    policy_id: str = "N/A"
    group_number: str = "N/A"
    relationship_to_insured: str = "N/A"


@dataclass
class SurgeryTypeOptions:
    """Structured surgery type options."""
    marked_options: List[str]
    all_available: List[str]


@dataclass
class ProviderRoles:
    """Structured provider role information."""
    ordering: str = "N/A"
    attending: str = "N/A"
    performing: str = "N/A"


@dataclass
class ArrowConnection:
    """Structured arrow connection information."""
    from_code: str
    to_code: str
    from_type: str  # "CPT" or "ICD"
    to_type: str    # "CPT" or "ICD"
    is_crossed_out: bool = False
    confidence: float = 1.0


@dataclass
class QuestionnaireItem:
    """Structured questionnaire item."""
    question_number: str
    question_text: str
    answer: str  # "Yes", "No", or "N/A"
    codes: Dict[str, str]  # {"yes": "G9902", "no": "G9903"}
    is_checked: bool = False


class OpenAIMedicalDocumentExtractor:
    """
    Enhanced medical document extraction service using OpenAI GPT-4 Vision.
    
    Optimized for detecting:
    - Hand-drawn circles around text
    - Checkmarks and X marks
    - Arrow connections between codes
    - Crossed-out connections
    - Handwritten annotations
    """
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gpt-4o"):
        """
        Initialize the medical document extractor.
        
        Args:
            api_key: OpenAI API key. If None, will use hardcoded key.
            model_name: OpenAI model name (default: gpt-4o)
        """
        self.logger = self._setup_logging()
        
        # Hardcoded API key as requested
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or ""
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key)
        self.model_name = model_name
        
        self.logger.info(f"Enhanced Medical Document Extractor initialized with model: {model_name}")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup professional logging configuration."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _convert_pdf_to_images(self, pdf_path: str) -> List[Image.Image]:
        """Convert PDF pages to high-quality images for processing."""
        self.logger.info("Converting PDF to high-resolution images...")
        
        # Higher DPI for better handwriting detection
        images = convert_from_path(pdf_path, dpi=400, fmt='PNG')
        self.logger.info(f"Converted {len(images)} pages at 400 DPI")
        
        return images
    
    def _enhance_image_contrast(self, image: Image.Image) -> Image.Image:
        """Enhance image contrast for better handwriting detection."""
        # Convert to numpy array
        img_array = np.array(image)
        
        # Apply contrast enhancement
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Contrast(image)
        enhanced = enhancer.enhance(1.5)  # Increase contrast by 50%
        
        return enhanced
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        buffer = io.BytesIO()
        # Use high quality PNG encoding
        image.save(buffer, format='PNG', optimize=False, quality=100)
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return img_str
    
    def _extract_page_1_data(self, image: Image.Image) -> Dict[str, Any]:
        """Extract marked procedures and circled items from page 1 (pink/salmon colored form)."""
        
        # Enhance image for better detection
        enhanced_image = self._enhance_image_contrast(image)
        
        prompt = """You are analyzing a PINK/SALMON colored medical fee ticket (Page 1).

CRITICAL VISUAL DETECTION TASKS:

1. SURGERY TYPE OPTIONS (Top right section):
Look at the row with these options: Est, New, Fol, Mohs, Surg, BL, Cos, TH
DETECT: Which options have HAND-DRAWN CIRCLES around them?
- A circle appears as a curved line drawn around the text
- Look for pen/pencil marks that encircle the text

2. LOCATION OF APPOINTMENT (Middle section):
Find the "Location of Appt:" line with options: ADR, ANG, BC, HIL, JXN, KZO, STU, STJ
DETECT: Which location code has a CIRCLE drawn around it?

3. PROVIDER INFORMATION:
Look for three labeled rows:
- "Ordering" row: Look for circled initials
- "Attending" row: Look for circled initials  
- "Performing" row: Look for circled initials
Each row has multiple doctor initials - identify which are circled

4. PATIENT HEADER INFORMATION:
Extract the typed/printed information:
- Patient name (after "Patient:")
- DOB (Date of Birth)
- MRN (Medical Record Number)
- Insurance information
- Provider name
- Appointment date

VISUAL DETECTION RULES:
- CIRCLED = hand-drawn circle, oval, or loop around text
- Circles are typically drawn in blue or black pen
- Circles may be imperfect/wobbly
- Only report items with visible circles
- If no circle is visible, mark as "none"

Return a JSON object with EXACTLY this structure:
{
  "patient_info": {
    "name": "extracted patient name",
    "dob": "extracted DOB",
    "mrn": "extracted MRN",
    "insurance": "insurance name",
    "provider": "provider initials",
    "appointment_date": "date"
  },
  "surgery_types_circled": ["list of circled options from Est/New/Fol/Mohs/Surg/BL/Cos/TH"],
  "location_circled": "the circled location code",
  "providers_circled": {
    "ordering": "circled initials or none",
    "attending": "circled initials or none",
    "performing": "circled initials or none"
  },
  "confidence_scores": {
    "surgery_detection": 0.0-1.0,
    "location_detection": 0.0-1.0,
    "provider_detection": 0.0-1.0
  }
}"""
        
        return self._call_openai_api(prompt, enhanced_image)
    
    def _extract_page_2_data(self, image: Image.Image) -> Dict[str, Any]:
        """Extract arrow connections from page 2 (pink coding reference sheet)."""
        
        enhanced_image = self._enhance_image_contrast(image)
        
        prompt = """You are analyzing a PINK medical coding reference sheet (Page 2) with CPT and ICD codes.

CRITICAL ARROW DETECTION TASK:

1. IDENTIFY CIRCLED CODES:
- Look for codes with hand-drawn circles around them
- Circles appear as pen/pencil loops around the text
- Both CPT codes (5-digit numbers) and ICD codes (letter+numbers) can be circled

2. TRACE ARROW CONNECTIONS:
- Find HAND-DRAWN LINES/ARROWS connecting circled codes
- These appear as pen-drawn lines between codes
- Follow the line from start to end

3. DETECT CROSSED-OUT CONNECTIONS:
IMPORTANT: Some arrows may be CROSSED OUT with zigzag or X patterns
- If a line has scribbles/crosses over it, it's CANCELLED
- Only report ACTIVE connections (not crossed out)

4. VISUAL ANALYSIS OF THE PROVIDED IMAGE:
Looking at the image, I can see:
- There's a drawn arrow/line connecting codes
- The line appears to connect from one code to another
- Check if the line has any cross-out marks

SPECIFIC AREAS TO CHECK:
- Left side: CPT procedure codes
- Right side: ICD diagnosis codes
- Middle sections: Additional code tables

EXTRACTION RULES:
- Only extract connections with visible drawn lines
- Ignore any crossed-out/scribbled connections
- Read code numbers carefully and completely
- Include code prefixes 

Return a JSON object with this structure:
{
  "circled_codes": [
    {
      "code": "exact code",
      "type": "CPT or ICD",
      "location": "brief description of position on page"
    }
  ],
  "arrow_connections": [
    {
      "from_code": "starting code",
      "from_type": "CPT or ICD",
      "to_code": "ending code", 
      "to_type": "CPT or ICD",
      "is_crossed_out": false,
      "line_style": "straight/curved/zigzag",
      "confidence": 0.0-1.0
    }
  ],
  "detection_notes": "any relevant observations about the markings"
}"""
        
        return self._call_openai_api(prompt, enhanced_image)
    
    def _extract_page_3_data(self, image: Image.Image) -> Dict[str, Any]:
        """Extract patient questionnaire responses from page 3 (purple/lavender form)."""
        
        enhanced_image = self._enhance_image_contrast(image)
        
        prompt = """You are analyzing a PURPLE/LAVENDER patient questionnaire form (Page 3).

CRITICAL CHECKMARK DETECTION TASKS:

1. PATIENT HEADER (Top section):
Extract the handwritten information:
- Patient name (may be printed and handwritten)
- MI (Middle Initial)
- Age (handwritten number)
- Date (handwritten)
- MRN/Account numbers

2. QUESTIONNAIRE CHECKMARKS - EXTREMELY IMPORTANT:
For EACH numbered question (#1, #2, #3, #4):

Question #1 - Tobacco Use:
- Look for CHECKMARK or X near "Yes (G9902)" or "No (G9903)"
- A checkmark appears as ✓ or / mark in pen

Question #2 - Health Care Proxy:
- Look for CHECKMARK near "Yes (1124F)" or "No (1123F with 8P modifier)"
- Check appears as pen mark near the option

Question #3 - High-risk Medications:
- Look for CHECKMARK near either option
- May have check near "NOT ordered (G9368)" or "ordered (G9367)"

Question #4 - Vaccinations (multiple sub-questions):
Each vaccine question has Yes/No options with codes:
- Meningococcal: Yes (G9414) / No (G9415)
- Tetanus/diphtheria: Yes (G9416) / No (G9417)  
- HPV: Yes (G9762) / No (G9763)

3. VISUAL CHECKMARK DETECTION:
- Checkmarks appear as ✓, /, or X marks
- Usually drawn in blue or black pen
- Located in space before or after the option text
- May be in a checkbox or just near the text

4. PROVIDER SIGNATURE:
- Look for handwritten signature at bottom
- Date near signature

CRITICAL INSTRUCTIONS:
- For each question, identify which option (Yes/No) has a visible checkmark
- If no checkmark is visible, mark as "not_answered"
- Read the G-codes and other codes carefully
- Note any handwritten additions or notes

Return a JSON with this structure:
{
  "patient_header": {
    "name": "patient name",
    "middle_initial": "MI",
    "age": "age number",
    "date": "date",
    "mrn": "MRN if visible"
  },
  "questionnaire_responses": [
    {
      "question_number": "1",
      "question_topic": "tobacco use",
      "selected_answer": "Yes or No or not_answered",
      "selected_code": "G9902 or G9903 or N/A",
      "has_visible_checkmark": true/false
    },
    {
      "question_number": "2", 
      "question_topic": "health care proxy",
      "selected_answer": "Yes or No or not_answered",
      "selected_code": "1124F or 1123F or N/A",
      "has_visible_checkmark": true/false
    },
    {
      "question_number": "3",
      "question_topic": "high-risk medications",
      "selected_answer": "NOT ordered or ordered or not_answered",
      "selected_code": "G9368 or G9367 or N/A",
      "has_visible_checkmark": true/false
    },
    {
      "question_number": "4a",
      "question_topic": "meningococcal vaccine",
      "selected_answer": "Yes or No or not_answered",
      "selected_code": "G9414 or G9415 or N/A",
      "has_visible_checkmark": true/false
    },
    {
      "question_number": "4b",
      "question_topic": "tetanus vaccine",
      "selected_answer": "Yes or No or not_answered",
      "selected_code": "G9416 or G9417 or N/A",
      "has_visible_checkmark": true/false
    },
    {
      "question_number": "4c",
      "question_topic": "HPV vaccine",
      "selected_answer": "Yes or No or not_answered",
      "selected_code": "G9762 or G9763 or N/A",
      "has_visible_checkmark": true/false
    }
  ],
  "provider_signature": {
    "has_signature": true/false,
    "date": "signature date if visible"
  },
  "detection_confidence": 0.0-1.0
}"""
        
        return self._call_openai_api(prompt, enhanced_image)
    
    def _call_openai_api(self, prompt: str, image: Image.Image) -> Dict[str, Any]:
        """Call OpenAI API with optimized parameters for medical form analysis."""
        try:
            # Convert image to base64
            image_base64 = self._image_to_base64(image)
            
            # Optimized model parameters for visual detection
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert medical form analyzer specializing in detecting handwritten marks, circles, checkmarks, and arrows on medical documents. Focus on precise visual detection of pen/pencil marks."
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_base64}",
                                    "detail": "high"  # Maximum detail for handwriting detection
                                }
                            }
                        ]
                    }
                ],
                max_tokens=4000,
                temperature=0.1,      # Low temperature for consistency
                top_p=0.95,          # Slightly higher for better detection
                frequency_penalty=0.0,
                presence_penalty=0.0,
                seed=42              # For reproducibility
            )
            
            result_text = response.choices[0].message.content
            
            # Extract and parse JSON
            json_str = self._extract_json_from_response(result_text)
            extracted_result = json.loads(json_str)
            
            self.logger.debug(f"Successfully extracted data from image")
            return extracted_result
            
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decode error: {str(e)}")
            # Return a structured error response
            return {
                "error": f"JSON decode error: {str(e)}",
                "raw_response": result_text if 'result_text' in locals() else "No response"
            }
        except Exception as e:
            self.logger.error(f"OpenAI API call failed: {str(e)}")
            return {"error": f"API call failed: {str(e)}"}
    
    def _extract_json_from_response(self, response_text: str) -> str:
        """Extract JSON string from OpenAI response."""
        # Try multiple extraction methods
        if "```json" in response_text:
            return response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            return response_text.split("```")[1].split("```")[0].strip()
        else:
            # Try to find JSON pattern
            import re
            json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
            matches = re.findall(json_pattern, response_text, re.DOTALL)
            if matches:
                # Return the longest match (likely the complete JSON)
                return max(matches, key=len)
            return response_text.strip()
    
    def extract_from_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract medical information from PDF document with enhanced accuracy.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dict: Structured extraction results
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        self.logger.info(f"Starting enhanced extraction from: {pdf_path}")
        
        try:
            # Convert PDF to high-quality images
            images = self._convert_pdf_to_images(pdf_path)
            
            # Process each page based on its type
            all_results = {
                "extraction_timestamp": datetime.now(timezone.utc).isoformat(),
                "document": pdf_path,
                "total_pages": len(images),
                "pages": []
            }
            
            for i, image in enumerate(images):
                page_num = i + 1
                self.logger.info(f"Processing page {page_num}/{len(images)}...")
                
                # Determine page type based on position in 3-page cycle
                page_type_in_set = ((page_num - 1) % 3) + 1
                
                try:
                    if page_type_in_set == 1:
                        # Pink fee ticket with surgery options
                        page_data = self._extract_page_1_data(image)
                        page_type = "fee_ticket"
                    elif page_type_in_set == 2:
                        # Pink coding reference with arrows
                        page_data = self._extract_page_2_data(image)
                        page_type = "coding_reference"
                    elif page_type_in_set == 3:
                        # Purple questionnaire
                        page_data = self._extract_page_3_data(image)
                        page_type = "questionnaire"
                    
                    all_results["pages"].append({
                        "page_number": page_num,
                        "page_type": page_type,
                        "data": page_data,
                        "extraction_success": "error" not in page_data
                    })
                    
                    self.logger.info(f"  ✓ Successfully processed page {page_num} ({page_type})")
                    
                except Exception as e:
                    self.logger.error(f"  ✗ Failed to process page {page_num}: {str(e)}")
                    all_results["pages"].append({
                        "page_number": page_num,
                        "page_type": "unknown",
                        "data": {"error": str(e)},
                        "extraction_success": False
                    })
            
            return all_results
            
        except Exception as e:
            self.logger.error(f"Extraction failed: {str(e)}")
            raise ValueError(f"Extraction failed: {str(e)}")
    
    def save_results(self, results: Dict[str, Any], output_path: str) -> None:
        """Save extraction results to JSON file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        self.logger.info(f"Results saved to: {output_path}")
    
    def print_summary(self, results: Dict[str, Any]) -> None:
        """Print a formatted summary of extraction results."""
        print("\n" + "="*80)
        print("ENHANCED MEDICAL DOCUMENT EXTRACTION SUMMARY")
        print("="*80)
        print(f"Document: {results['document']}")
        print(f"Timestamp: {results['extraction_timestamp']}")
        print(f"Total Pages: {results['total_pages']}")
        print()
        
        for page in results["pages"]:
            print(f"\n{'='*60}")
            print(f"PAGE {page['page_number']} - {page['page_type'].upper()}")
            print(f"{'='*60}")
            
            if not page["extraction_success"]:
                print(f"  ✗ Extraction failed: {page['data'].get('error', 'Unknown error')}")
                continue
            
            data = page["data"]
            
            if page["page_type"] == "fee_ticket":
                # Page 1 summary
                if "patient_info" in data:
                    print("\nPATIENT INFORMATION:")
                    info = data["patient_info"]
                    print(f"  • Name: {info.get('name', 'N/A')}")
                    print(f"  • DOB: {info.get('dob', 'N/A')}")
                    print(f"  • MRN: {info.get('mrn', 'N/A')}")
                
                if "surgery_types_circled" in data:
                    print(f"\nCIRCLED SURGERY TYPES:")
                    for item in data["surgery_types_circled"]:
                        print(f"  ✓ {item}")
                
                if "location_circled" in data:
                    print(f"\nCIRCLED LOCATION: {data['location_circled']}")
                
                if "providers_circled" in data:
                    print(f"\nCIRCLED PROVIDERS:")
                    providers = data["providers_circled"]
                    for role, initials in providers.items():
                        if initials and initials != "none":
                            print(f"  • {role.capitalize()}: {initials}")
            
            elif page["page_type"] == "coding_reference":
                # Page 2 summary
                if "circled_codes" in data:
                    print(f"\nCIRCLED CODES ({len(data['circled_codes'])}):")
                    for code_info in data["circled_codes"]:
                        print(f"  ✓ {code_info['code']} ({code_info['type']})")
                
                if "arrow_connections" in data:
                    print(f"\nARROW CONNECTIONS:")
                    for conn in data["arrow_connections"]:
                        if not conn.get("is_crossed_out", False):
                            print(f"  → {conn['from_code']} ({conn['from_type']}) ——→ {conn['to_code']} ({conn['to_type']})")
                            print(f"    Confidence: {conn.get('confidence', 0)*100:.0f}%")
            
            elif page["page_type"] == "questionnaire":
                # Page 3 summary
                if "patient_header" in data:
                    print("\nPATIENT HEADER:")
                    header = data["patient_header"]
                    print(f"  • Name: {header.get('name', 'N/A')}")
                    print(f"  • Age: {header.get('age', 'N/A')}")
                    print(f"  • Date: {header.get('date', 'N/A')}")
                
                if "questionnaire_responses" in data:
                    print("\nQUESTIONNAIRE RESPONSES:")
                    for item in data["questionnaire_responses"]:
                        if item.get("has_visible_checkmark", False):
                            print(f"  Question #{item['question_number']}: {item['question_topic']}")
                            print(f"    ✓ {item['selected_answer']} (Code: {item['selected_code']})")
        
        print("\n" + "="*80)


def main():
    """Main execution function with enhanced extraction."""
    try:
        # Initialize enhanced extractor
        extractor = OpenAIMedicalDocumentExtractor(model_name="gpt-4o")
        
        # Your PDF file
        pdf_file = "set1.pdf"  # Change this to your PDF filename
        
        print(f"\n🔍 Starting enhanced extraction of: {pdf_file}")
        print("This may take a moment for high-accuracy visual analysis...")
        
        # Extract with enhanced accuracy
        results = extractor.extract_from_pdf(pdf_file)
        
        # Save results
        output_file = f"enhanced_extraction_{pdf_file.replace('.pdf', '')}.json"
        extractor.save_results(results, output_file)
        
        # Print summary
        extractor.print_summary(results)
        
        print(f"\n✅ Extraction complete! Results saved to: {output_file}")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Ensure the PDF file exists in the current directory")
        print("2. Check that all required packages are installed:")
        print("   pip install openai pdf2image pillow numpy")
        print("3. Verify your OpenAI API key has GPT-4 Vision access")


if __name__ == "__main__":
    main()