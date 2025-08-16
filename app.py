import io
import fitz  # PyMuPDF
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from paddleocr import PaddleOCR
from typing import List
import numpy as np
import cv2, os
import logging, datetime
from fastapi import Depends
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Use absolute path to avoid surprises
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_URL = f"sqlite:///{os.path.join(BASE_DIR, 'invoices.db')}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class Invoice(Base):
    __tablename__ = "invoices"
    id = Column(Integer, primary_key=True, index=True)
    patient_name = Column(String, nullable=True)
    invoice_date = Column(String, nullable=True)  
    total_amount = Column(Float, nullable=True)
    itemized_services = Column(JSON, nullable=True)  
    handwritten_notes = Column(String, nullable=True)
    analysis_results = Column(JSON, nullable=True)  
    raw_text = Column(String, nullable=True)  
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    user_id = Column(String, nullable=True)

# Create tables
Base.metadata.create_all(bind=engine)

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()



# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------
# PaddleOCR Initialization (Simple)
# -----------------------------
try:
    # Simple initialization - let PaddleOCR use defaults
    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    print("✅ PaddleOCR initialized successfully")
except Exception as e:
    print(f"❌ Failed to initialize PaddleOCR: {e}")
    raise

app = FastAPI(
    title="Simple PaddleOCR API",
    description="Extracts text from images and PDFs using PaddleOCR",
    version="1.0.0"
)

# -----------------------------
# Helper Functions
# -----------------------------

def extract_text_from_image_bytes(image_bytes: bytes) -> List[dict]:
    """Run OCR on an image in memory."""
    try:
        print(f"📷 Processing image of size: {len(image_bytes)} bytes")
        
        # Decode image from bytes
        np_img = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        
        if img is None:
            print("❌ Failed to decode image")
            return []
        
        print(f"📐 Image dimensions: {img.shape}")
        
        # Run OCR
        print("🔍 Running OCR...")
        results = ocr.ocr(img)
        
        print(f"📝 Raw OCR results: {results}")
        
        # Parse results - handle the new PaddleOCR format
        extracted = []
        if results and len(results) > 0:
            result = results[0]  # Get first result
            
            # Check if it's the new dictionary format
            if isinstance(result, dict):
                # New format: dictionary with 'rec_texts', 'rec_scores', 'rec_polys'
                rec_texts = result.get('rec_texts', [])
                rec_scores = result.get('rec_scores', [])
                rec_polys = result.get('rec_polys', [])
                
                print(f"📝 Found {len(rec_texts)} text items in new format")
                
                for i, text in enumerate(rec_texts):
                    if text and text.strip():  # Only non-empty text
                        confidence = rec_scores[i] if i < len(rec_scores) else 0.0
                        bbox = rec_polys[i].tolist() if i < len(rec_polys) else []
                        
                        extracted.append({
                            "text": text.strip(),
                            "confidence": round(float(confidence), 3),
                            "bbox": bbox
                        })
            else:
                # Old format: list of [bbox, (text, confidence)]
                print("📝 Using old format parsing")
                for line in result:
                    try:
                        bbox, (text, confidence) = line
                        if text and text.strip():
                            extracted.append({
                                "text": text.strip(),
                                "confidence": round(confidence, 3),
                                "bbox": bbox
                            })
                    except Exception as e:
                        print(f"⚠️  Error parsing line: {e}")
                        continue
        
        print(f"✅ Extracted {len(extracted)} text items")
        return extracted
        
    except Exception as e:
        print(f"❌ Error in OCR processing: {e}")
        return []

def extract_text_from_pdf(file_bytes: bytes) -> List[dict]:
    """Convert PDF to images and run OCR."""
    try:
        print(f"📄 Processing PDF of size: {len(file_bytes)} bytes")
        
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        all_text = []
        
        print(f"📚 PDF has {len(doc)} pages")
        
        for page_num in range(len(doc)):
            print(f"📖 Processing page {page_num + 1}")
            
            # Convert page to image
            page = doc[page_num]
            pix = page.get_pixmap(dpi=200)  # Lower DPI for faster processing
            img_bytes = pix.tobytes("png")
            
            # Run OCR on page image
            page_text = extract_text_from_image_bytes(img_bytes)
            all_text.extend(page_text)
        
        doc.close()
        print(f"✅ Total extracted from PDF: {len(all_text)} text items")
        return all_text
        
    except Exception as e:
        print(f"❌ Error processing PDF: {e}")
        return []

# -----------------------------
# API Endpoints
# -----------------------------

@app.post("/invoices")
def create_invoice(invoice: dict, db: Session = Depends(get_db)):
    new_invoice = Invoice(
        patient_name=invoice.get("patient_name"),
        invoice_date=invoice.get("invoice_date"),
        total_amount=invoice.get("total_amount"),
        itemized_services=invoice.get("itemized_services"),
        handwritten_notes=invoice.get("handwritten_notes"),
        analysis_results=invoice.get("analysis_results"),
        raw_text=invoice.get("raw_text"),
        user_id=invoice.get("user_id")
    )
    db.add(new_invoice)
    db.commit()
    db.refresh(new_invoice)
    return {"id": new_invoice.id, "message": "Invoice stored successfully"}

@app.get("/invoices")
def get_invoices(patient_name: str = None, start_date: str = None, end_date: str = None, db: Session = Depends(get_db)):
    query = db.query(Invoice)
    if patient_name:
        query = query.filter(Invoice.patient_name.contains(patient_name))
    if start_date and end_date:
        query = query.filter(Invoice.invoice_date.between(start_date, end_date))
    print(query.all())
    return query.all()

@app.get("/invoices/{invoice_id}")
def get_invoice(invoice_id: int, db: Session = Depends(get_db)):
    invoice = db.query(Invoice).filter(Invoice.id == invoice_id).first()
    print(invoice)
    if not invoice:
        raise HTTPException(status_code=404, detail="Invoice not found")
    return invoice

@app.put("/invoices/{invoice_id}")
def update_invoice(invoice_id: int, updated: dict, db: Session = Depends(get_db)):
    invoice = db.query(Invoice).filter(Invoice.id == invoice_id).first()
    if not invoice:
        raise HTTPException(status_code=404, detail="Invoice not found")
    for key, value in updated.items():
        setattr(invoice, key, value)
    db.commit()
    return {"message": "Invoice updated"}

@app.delete("/invoices/{invoice_id}")
def delete_invoice(invoice_id: int, db: Session = Depends(get_db)):
    invoice = db.query(Invoice).filter(Invoice.id == invoice_id).first()
    if not invoice:
        raise HTTPException(status_code=404, detail="Invoice not found")
    db.delete(invoice)
    db.commit()
    return {"message": "Invoice deleted"}

@app.post("/extract-for-llm")
async def extract_for_llm(file: UploadFile = File(...)):
    """Extract text optimized for LLM processing - returns clean, structured text."""
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    filename = file.filename.lower()
    print(f"📁 Processing file for LLM: {file.filename}")
    
    # Check file type
    allowed_extensions = ['.jpg', '.jpeg', '.png', '.pdf']
    if not any(filename.endswith(ext) for ext in allowed_extensions):
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Allowed: {allowed_extensions}"
        )
    
    try:
        # Read file content
        file_bytes = await file.read()
        
        if len(file_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty file")
        
        # Process based on file type
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            text_data = extract_text_from_image_bytes(file_bytes)
        elif filename.endswith('.pdf'):
            text_data = extract_text_from_pdf(file_bytes)
        else:
            text_data = []
        
        if text_data:
            # Create clean text for LLM
            all_text = [item["text"] for item in text_data]
            
            # Clean and structure the text better
            clean_text = []
            for text in all_text:
                # Basic cleaning
                cleaned = text.strip()
                if cleaned and len(cleaned) > 1:  # Skip single characters and empty
                    clean_text.append(cleaned)
            
            # Join with spaces and clean up
            full_text = " ".join(clean_text)
            
            # Simple formatting improvements
            full_text = full_text.replace("  ", " ")  # Remove double spaces
            
            # Return LLM-optimized format
            return JSONResponse(content={
                "document_text": full_text,
                "ready_for_analysis": True,
                "text_quality": "ocr_extracted",
                "word_count": len(full_text.split()),
                "source_file": file.filename
            })
        else:
            return JSONResponse(content={
                "document_text": "",
                "ready_for_analysis": False,
                "error": "No text could be extracted from the document",
                "source_file": file.filename
            })
            
    except Exception as e:
        print(f"❌ API Error: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.post("/extract")
async def extract_text(file: UploadFile = File(...)):
    """Extract text from uploaded image or PDF file."""
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    filename = file.filename.lower()
    print(f"📁 Processing file: {file.filename}")
    
    # Check file type
    allowed_extensions = ['.jpg', '.jpeg', '.png', '.pdf']
    if not any(filename.endswith(ext) for ext in allowed_extensions):
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Allowed: {allowed_extensions}"
        )
    
    try:
        # Read file content
        file_bytes = await file.read()
        
        if len(file_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty file")
        
        # Process based on file type
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            text_data = extract_text_from_image_bytes(file_bytes)
        elif filename.endswith('.pdf'):
            text_data = extract_text_from_pdf(file_bytes)
        else:
            text_data = []
        
        # Prepare response
        response = {
            "filename": file.filename,
            "total_items": len(text_data),
            "extracted_text": text_data
        }
        
        if not text_data:
            response["message"] = "No text found in the file"
            response["suggestions"] = [
                "Ensure the image contains clear, readable text",
                "Try a higher resolution image",
                "Check if the text is in English (currently configured for 'en')"
            ]
        
        return JSONResponse(content=response)
        
    except Exception as e:
        print(f"❌ API Error: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.get("/")
def root():
    """Root endpoint with API information."""
    return {
        "message": "Simple PaddleOCR API is running! 🚀",
        "endpoints": {
            "/extract": "POST - Extract text with detailed structure",
            "/extract-for-llm": "POST - Extract text optimized for LLM analysis (recommended)",
        },
        "usage": {
            "supported_files": ["JPG", "JPEG", "PNG", "PDF"],
            "examples": {
                "detailed_extraction": "curl -X POST -F 'file=@invoice.pdf' http://localhost:8000/extract",
                "llm_ready_extraction": "curl -X POST -F 'file=@invoice.pdf' http://localhost:8000/extract-for-llm"
            }
        },
        "recommendation": "Use /extract-for-llm for cleaner text suitable for LLM processing"
    }

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting PaddleOCR FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)