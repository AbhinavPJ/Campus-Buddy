#!/usr/bin/env python3
"""
Hybrid BM25 + BERT Embedding Generator
=====================================

This script processes PDF and TXT files to create embeddings for hybrid retrieval.
Run this script ONCE to generate embeddings, then use the Streamlit app for querying.

Usage:
    python generate_embeddings.py

Requirements:
    pip install torch transformers PyMuPDF pytesseract pillow tqdm rank-bm25

File Structure:
    pdfs/                    # Input folder with PDF/TXT files
    ├── document1.pdf
    ├── document2.txt
    └── ...
    
Output Files:
    hybrid_embeddings.pt     # BERT embeddings and passages
    bm25_index.pkl          # BM25 index and keyword statistics
"""

import os
import fitz  # PyMuPDF
import torch
from transformers import BertTokenizer, BertModel
import pytesseract
from PIL import Image
import io
from tqdm import tqdm
import time
import pickle
from rank_bm25 import BM25Okapi
import re
from collections import Counter
import sys

def load_models():
    """Load BERT model and tokenizer"""
    print("🤖 Loading BERT model and tokenizer...")
    try:
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertModel.from_pretrained("bert-base-uncased")
        model.eval()  # Set to evaluation mode
        print("✅ BERT model loaded successfully!")
        return tokenizer, model
    except Exception as e:
        print(f"❌ Error loading BERT model: {e}")
        print("💡 Try: pip install torch transformers")
        sys.exit(1)

def preprocess_text_for_bm25(text):
    """Preprocess text for BM25 - preserve important keywords like PYL101, APL100"""
    # Convert to lowercase but preserve alphanumeric patterns (course codes)
    text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuation but keep alphanumeric
    tokens = text.lower().split()
    
    # Filter out very short tokens but keep course codes
    filtered_tokens = []
    for token in tokens:
        if len(token) >= 2 or re.match(r'[a-zA-Z]+\d+', token):  # Keep if length >= 2 or matches course code pattern
            filtered_tokens.append(token)
    
    return filtered_tokens

def extract_important_keywords(text):
    """Extract important keywords like course codes, technical terms"""
    # Pattern for course codes like PYL101, APL100, CSE101, etc.
    course_codes = re.findall(r'[A-Z]{2,4}\d{3,4}', text)
    
    # Pattern for technical terms (capitalized words)
    technical_terms = re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)*\b', text)
    
    # Pattern for acronyms
    acronyms = re.findall(r'\b[A-Z]{2,}\b', text)
    
    keywords = course_codes + technical_terms + acronyms
    return list(set(keywords))  # Remove duplicates

def split_into_passages(text, max_tokens=512, tokenizer=None):
    """Split text into passages that fit within BERT's token limit"""
    sentences = text.split('. ')
    passages, current_passage = [], ""
    
    print(f"📝 Splitting text into passages (max {max_tokens} tokens each)...")
    
    for sentence in sentences:
        # Test if adding this sentence would exceed token limit
        test_passage = current_passage + sentence + ". "
        tokens = tokenizer.encode(test_passage, add_special_tokens=True)
        
        if len(tokens) < max_tokens:
            current_passage += sentence + ". "
        else:
            if current_passage.strip():
                passages.append(current_passage.strip())
                print(f"  ✓ Created passage with {len(tokenizer.encode(current_passage.strip()))} tokens")
            current_passage = sentence + ". "
    
    # Add the last passage if it exists
    if current_passage.strip():
        passages.append(current_passage.strip())
        print(f"  ✓ Created final passage with {len(tokenizer.encode(current_passage.strip()))} tokens")
    
    print(f"📄 Total passages created: {len(passages)}")
    return passages

def extract_text_from_pdf(file_path):
    """Extract text from PDF with OCR fallback and debug info"""
    print(f"📖 Opening PDF: {file_path}")
    try:
        doc = fitz.open(file_path)
    except Exception as e:
        print(f"❌ Error opening PDF: {e}")
        return ""
    
    all_text = []
    total_pages = len(doc)
    print(f"📄 PDF has {total_pages} pages")
    
    for page_num, page in enumerate(doc):
        print(f"  Processing page {page_num + 1}/{total_pages}...")
        
        # Try to extract text directly
        text = page.get_text()
        
        if text.strip():  # If text is detected, use it
            print(f"    ✓ Extracted {len(text)} characters from text layer")
            all_text.append(text)
        else:  # Use OCR if page is likely an image
            print(f"    📷 No text layer found, using OCR...")
            try:
                pix = page.get_pixmap(dpi=300)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                ocr_text = pytesseract.image_to_string(img)
                print(f"    ✓ OCR extracted {len(ocr_text)} characters")
                all_text.append(ocr_text)
            except Exception as e:
                print(f"    ❌ OCR failed: {e}")
                all_text.append("")
    
    doc.close()
    combined_text = "\n".join(all_text)
    print(f"📝 Total text extracted: {len(combined_text)} characters")
    return combined_text

def get_bert_embedding(text, tokenizer, model):
    """Get BERT embedding for a text passage"""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding="max_length"
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
        # Use the [CLS] token embedding as the passage representation
        embedding = outputs.last_hidden_state[:, 0, :].squeeze(0)
    
    return embedding

def process_files(pdf_folder="pdfs"):
    """Main function to process all files and generate embeddings"""
    
    # Load models
    tokenizer, model = load_models()
    
    # Initialize storage
    all_passages = []
    all_embeddings = []
    all_sources = []
    all_bm25_tokens = []  # For BM25 corpus
    all_keywords = []     # Store extracted keywords per passage

    print(f"📁 Looking for files in '{pdf_folder}' directory...")

    # Check if folder exists
    if not os.path.exists(pdf_folder):
        print(f"❌ Error: '{pdf_folder}' directory not found!")
        print(f"💡 Please create a '{pdf_folder}' directory and add your PDF/TXT files")
        return False

    # Get list of files
    files = [f for f in os.listdir(pdf_folder) if f.lower().endswith(('.pdf', '.txt'))]
    print(f"📂 Found {len(files)} files to process: {files}")

    if not files:
        print("❌ No PDF or TXT files found in the directory!")
        print(f"💡 Please add PDF or TXT files to the '{pdf_folder}' directory")
        return False

    # Process each file
    for file_idx, filename in enumerate(files):
        print(f"\n🔄 Processing file {file_idx + 1}/{len(files)}: {filename}")
        print("=" * 50)
        
        file_path = os.path.join(pdf_folder, filename)
        
        try:
            # Extract text depending on file type
            if filename.lower().endswith(".pdf"):
                full_text = extract_text_from_pdf(file_path)
            else:
                print(f"📄 Reading TXT file: {filename}")
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        full_text = f.read()
                    print(f"📝 Read {len(full_text)} characters from TXT file")
                except Exception as e:
                    print(f"❌ Error reading TXT file: {e}")
                    continue
            
            # Check if we extracted any meaningful text
            if not full_text.strip():
                print(f"⚠️ Warning: No text extracted from {filename}")
                continue
            
            # Split into passages
            passages = split_into_passages(full_text, tokenizer=tokenizer)
            
            if not passages:
                print(f"⚠️ Warning: No passages created from {filename}")
                continue
            
            print(f"🧠 Processing {len(passages)} passages for hybrid retrieval...")
            
            # Generate embeddings and BM25 tokens with progress bar
            for passage_idx, passage in enumerate(tqdm(passages, desc=f"Processing {filename}")):
                try:
                    # Get BERT embedding
                    embedding = get_bert_embedding(passage, tokenizer, model)
                    
                    # Get BM25 tokens
                    bm25_tokens = preprocess_text_for_bm25(passage)
                    
                    # Extract important keywords
                    keywords = extract_important_keywords(passage)
                    
                    # Store everything
                    all_passages.append(passage)
                    all_embeddings.append(embedding)
                    all_sources.append(filename)
                    all_bm25_tokens.append(bm25_tokens)
                    all_keywords.append(keywords)
                    
                    # Debug info every 10 passages
                    if (passage_idx + 1) % 10 == 0:
                        print(f"    ✓ Processed {passage_idx + 1}/{len(passages)} passages")
                        if keywords:
                            print(f"    🔑 Keywords found: {keywords[:5]}...")  # Show first 5 keywords
                        
                except Exception as e:
                    print(f"    ❌ Error processing passage {passage_idx + 1}: {e}")
                    continue
            
            print(f"✅ Successfully processed {filename}")
            print(f"   📊 Added {len(passages)} passages to collection")
            
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")
            continue

    # Final summary
    print(f"\n📊 PROCESSING COMPLETE")
    print("=" * 50)
    print(f"📄 Total files processed: {len(set(all_sources))}")
    print(f"📝 Total passages created: {len(all_passages)}")
    print(f"🧠 Total embeddings generated: {len(all_embeddings)}")

    if not all_embeddings:
        print("❌ No embeddings were generated! Check your files and try again.")
        return False

    # Create BM25 index
    print(f"\n🔍 Creating BM25 index...")
    try:
        bm25 = BM25Okapi(all_bm25_tokens)
        print(f"✅ BM25 index created with {len(all_bm25_tokens)} documents")
    except Exception as e:
        print(f"❌ Error creating BM25 index: {e}")
        return False

    # Analyze keywords
    all_keywords_flat = [kw for kw_list in all_keywords for kw in kw_list]
    keyword_counts = Counter(all_keywords_flat)
    print(f"🔑 Found {len(keyword_counts)} unique keywords")
    if keyword_counts:
        print(f"   Top keywords: {dict(keyword_counts.most_common(10))}")

    # Save everything
    print(f"\n💾 Saving hybrid retrieval data...")
    try:
        # Save BERT embeddings
        torch.save({
            "passages": all_passages,
            "embeddings": torch.stack(all_embeddings),
            "sources": all_sources,
            "keywords": all_keywords
        }, "hybrid_embeddings.pt")
        
        # Save BM25 index and tokens
        with open("bm25_index.pkl", "wb") as f:
            pickle.dump({
                "bm25": bm25,
                "bm25_tokens": all_bm25_tokens,
                "keyword_counts": keyword_counts
            }, f)
        
        print("✅ Successfully saved hybrid retrieval data!")
        print("   📄 BERT embeddings saved to: hybrid_embeddings.pt")
        print("   🔍 BM25 index saved to: bm25_index.pkl")
        
        # Verify the saved files
        print(f"🔍 Verifying saved files...")
        bert_data = torch.load("hybrid_embeddings.pt")
        print(f"   ✓ BERT - Passages: {len(bert_data['passages'])}")
        print(f"   ✓ BERT - Embeddings shape: {bert_data['embeddings'].shape}")
        print(f"   ✓ BERT - Sources: {len(bert_data['sources'])}")
        
        with open("bm25_index.pkl", "rb") as f:
            bm25_data = pickle.load(f)
            print(f"   ✓ BM25 - Documents: {len(bm25_data['bm25_tokens'])}")
            print(f"   ✓ BM25 - Keywords: {len(bm25_data['keyword_counts'])}")
        
    except Exception as e:
        print(f"❌ Error saving data: {e}")
        return False

    print(f"\n🎉 Hybrid retrieval system ready!")
    print("🔍 BM25 will handle exact keyword matching (PYL101, APL100, etc.)")
    print("🧠 BERT will handle semantic similarity matching")
    print("\n💡 Next steps:")
    print("   1. Run: streamlit run chat_app.py")
    print("   2. Start asking questions about your documents!")
    
    return True

if __name__ == "__main__":
    print("🚀 Starting Hybrid Embedding Generation")
    print("=" * 50)
    
    # Check for required packages
    try:
        import torch
        import transformers
        import fitz
        import pytesseract
        from PIL import Image
        from tqdm import tqdm
        from rank_bm25 import BM25Okapi
        print("✅ All required packages found")
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("💡 Install with: pip install torch transformers PyMuPDF pytesseract pillow tqdm rank-bm25")
        sys.exit(1)
    
    # Process files
    success = process_files()
    
    if success:
        print("\n🎉 SUCCESS! Embeddings generated successfully!")
        print("Run 'streamlit run chat_app.py' to start the chat interface")
    else:
        print("\n❌ FAILED! Please check the errors above and try again")
        sys.exit(1)