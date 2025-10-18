import PyPDF2
from PIL import Image
import pytesseract
from docx import Document
import os

class DocumentProcessor:
    def __init__(self):
        # Try to set tesseract path for Windows
        if os.name == 'nt':  # Windows
            tesseract_path = r'C:/Program Files/Tesseract-OCR/tesseract.exe'
            if os.path.exists(tesseract_path):
                pytesseract.pytesseract.tesseract_cmd = tesseract_path
    
    def extract_text(self, filepath):
        """Extract text from various document formats"""
        file_extension = filepath.rsplit('.', 1)[1].lower()
        
        if file_extension == 'pdf':
            return self._extract_from_pdf(filepath)
        elif file_extension in ['png', 'jpg', 'jpeg']:
            return self._extract_from_image(filepath)
        elif file_extension == 'docx':
            return self._extract_from_docx(filepath)
        elif file_extension == 'txt':
            return self._extract_from_txt(filepath)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
    
    def _extract_from_pdf(self, filepath):
        """Extract text from PDF"""
        text = ""
        try:
            with open(filepath, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            
            # If no text extracted, try OCR
            if not text.strip():
                text = self._ocr_pdf(filepath)
            
            return text.strip()
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")
    
    def _extract_from_image(self, filepath):
        """Extract text from image using OCR"""
        try:
            image = Image.open(filepath)
            text = pytesseract.image_to_string(image)
            return text.strip()
        except Exception as e:
            raise Exception(f"Error extracting text from image: {str(e)}")
    
    def _extract_from_docx(self, filepath):
        """Extract text from Word document"""
        try:
            doc = Document(filepath)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text.strip()
        except Exception as e:
            raise Exception(f"Error extracting text from DOCX: {str(e)}")
    
    def _extract_from_txt(self, filepath):
        """Extract text from plain text file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                return file.read().strip()
        except Exception as e:
            raise Exception(f"Error extracting text from TXT: {str(e)}")
    
    def _ocr_pdf(self, filepath):
        """Perform OCR on PDF if text extraction fails"""
        try:
            from pdf2image import convert_from_path
            images = convert_from_path(filepath)
            text = ""
            for image in images:
                text += pytesseract.image_to_string(image) + "\n"
            return text.strip()
        except:
            return ""