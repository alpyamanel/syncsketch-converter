import streamlit as st
import fitz  # PyMuPDF
from PIL import Image
import io
import pandas as pd
import concurrent.futures
from google import genai
from google.genai import types
from pydantic import BaseModel
import json

st.set_page_config(page_title="SyncSketch PDF to Excel", layout="wide")

# Process 5 pages at a time simultaneously (Incredibly fast)
MAX_WORKERS = 5 

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
gemini_key = None
if "GEMINI_API_KEY" in st.secrets:
    gemini_key = st.secrets["GEMINI_API_KEY"]
    st.sidebar.success("🔒 System running securely via Cloud Secrets Manager.")
else:
    st.sidebar.title("Configuration")
    gemini_key = st.sidebar.text_input("Gemini API Key", type="password")

# ----------------------------------------------------------------------
# Core Logic: Extract Pure Images & Render Pages
# ----------------------------------------------------------------------
def process_pdf_pages(pdf_bytes):
    """
    Instead of blindly cutting the page into thirds, this extracts the 
    pure, original embedded screenshot files directly from the PDF.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages_data = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # 1. Extract pure embedded screenshot files
        screenshots_info = []
        for img_info in page.get_image_info(xrefs=True):
            xref = img_info.get("xref")
            if xref:
                try:
                    base_image = doc.extract_image(xref)
                    pil_img = Image.open(io.BytesIO(base_image["image"]))
                    # Filter out tiny icons/avatars; only keep the large video frames
                    if pil_img.width > 200 and pil_img.height > 100:
                        screenshots_info.append({
                            "y0": img_info["bbox"][1], 
                            "img": pil_img
                        })
                except Exception:
                    pass
                    
        # Sort top-to-bottom so they perfectly match the text read by Gemini
        screenshots_info.sort(key=lambda x: x["y0"])
        extracted_images = [s["img"] for s in screenshots_info]
        
        # 2. Render the full page for Gemini to read the text
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        full_page_img = Image.open(io.BytesIO(pix.tobytes("png")))
        
        # Safe Fallback just in case the PDF is flattened
        if not extracted_images:
            height = full_page_img.height
            row_height = height // 3
            for i in range(3):
                top = i * row_height
                bottom = (i + 1) * row_height
                row_img = full_page_img.crop((0, top, int(full_page_img.width * 0.55), bottom))
                extracted_images.append(row_img)
                
        pages_data.append({
            "page_img": full_page_img,
            "screenshots": extracted_images
        })
        
    return pages_data

# ----------------------------------------------------------------------
# Gemini Vision AI
# ----------------------------------------------------------------------
def extract_page_data(page_data, api_key):
    """Sends a single full page to Gemini to extract all rows at once."""
    page_img = page_data["page_img"]
    screenshots = page_data["screenshots"]
    
    client = genai.Client(api_key=api_key)
    bio = io.BytesIO()
    page_img.save(bio, format="PNG")
    
    prompt = f"""
    Analyze this page from a SyncSketch review PDF.
    There are exactly {len(screenshots)} review rows on this page.
    Extract the data for EACH row in top-to-bottom order, returning a list of exactly {len(screenshots)} objects.

    For each row, look at the screenshot on the left and the text on the right:
    1. 'Scene/File Identification': Found top-right INSIDE the left screenshot.
    2. 'Batch Timecode': Found bottom-centre near 'REC TC' INSIDE the left screenshot.
    3. 'Episode Timecode': Found bottom-right INSIDE the left screenshot.
    4. 'Name': The bold/black text before the colon (:) in the right text area.
    5. 'Note': The text after the colon (:) in the right text area.

    CRITICAL NAME NORMALIZATION & FIXES:
    SyncSketch OCR often cuts off or misspells reviewer names. YOU MUST FIX THESE:
    - If you see variations like 'Trevor Wa', 'Trever', 'Trever Wall', or 'Trevor W', output strictly 'Trevor Wall'.
    - If you see variations like 'James Ar', 'Jame', or 'James A', output strictly 'James Anderson'.
    - Apply logical correction to fix any obvious truncations (e.g., 'William F' to 'William Fung').
    - Ignore green text (like FRAME, TIMECODE). Mark missing/unclear strictly as [UNCLEAR].
    """
    
    class SyncSketchRow(BaseModel):
        scene_id: str
        batch_timecode: str
        episode_timecode: str
        reviewer_name: str
        reviewer_note: str

    class PageExtraction(BaseModel):
        rows: list[SyncSketchRow]

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[
                types.Part.from_bytes(data=bio.getvalue(), mime_type="image/png"),
                prompt
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=PageExtraction,
                temperature=0.0
            ),
        )
        
        data = json.loads(response.text)
        parsed_rows = data.get("rows", [])
        
        results = []
        max_len = max(len(parsed_rows), len(screenshots))
        
        for i in range(max_len):
            # Gracefully align the AI text data with the extracted images
            if i < len(parsed_rows):
                r = parsed_rows[i]
                sc_id = r.get("scene_id", "[UNCLEAR]")
                btc = r.get("batch_timecode", "[UNCLEAR]")
                etc = r.get("episode_timecode", "[UNCLEAR]")
                name = r.get("reviewer_name", "[UNCLEAR]")
                note = r.get("reviewer_note", "[UNCLEAR]")
            else:
                sc_id = btc = etc = name = note = "[UNCLEAR]"
                
            img = screenshots[i] if i < len(screenshots) else None
            
            results.append({
                "Scene/File Identification": sc_id,
                "Batch Timecode": btc,
                "Episode Timecode": etc,
                "Name": name,
                "Note": note,
                "left_img_obj": img
            })
        return results
        
    except Exception as e:
        return [{
            "Scene/File Identification": "[UNCLEAR]",
            "Batch Timecode": "[UNCLEAR]",
            "Episode Timecode": "[UNCLEAR]",
            "Name": "[UNCLEAR]",
            "Note": f"[ERROR: {str(e)}]",
            "left_img_obj": screenshots[0] if screenshots else None
        }]

def process_multiple_pdfs(uploaded_files, api_key):
    all_extracted_data = []
    
    for file in uploaded_files:
        pdf_bytes = file.read()
        pages_data = process_pdf_pages(pdf_bytes)
        
        progress_bar = st.progress(0, text=f"Analyzing {file.name} with AI...")
        
        page_results = [None] * len(pages_data)
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(extract_page_data, p_data, api_key): i for i, p_data in enumerate(pages_data)}
            for count, future in enumerate(concurrent.futures.as_completed(futures)):
                idx = futures[future]
                page_results[idx] = future.result()
                progress_bar.progress((count + 1) / len(pages_data))
        
        for pr in page_results:
            for row in pr:
                row["Source File"] = file.name
                all_extracted_data.append(row)
                
        progress_bar.empty()
        
    return all_extracted_data

# ----------------------------------------------------------------------
# Perfect Excel Generation
# ----------------------------------------------------------------------
def generate_excel_with_images(df):
    """Creates a beautifully formatted Excel file with perfect image boundaries."""
    output = io.BytesIO()
    
    text_df = df.drop(columns=['left_img_obj'], errors='ignore')
    text_df.insert(5, 'Visual Capture', '') 
    
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    text_df.to_excel(writer, sheet_name='SyncSketch Notes', index=False)
    
    workbook = writer.book
    worksheet = writer.sheets['SyncSketch Notes']
    
    # Beautiful formatting profiles
    header_format = workbook.add_format({
        'bold': True,
        'valign': 'vcenter',
        'align': 'center',
        'bg_color': '#D3D3D3',
        'border': 1
    })
    
    cell_format = workbook.add_format({
        'text_wrap': True,
        'valign': 'vcenter',
        'border': 1
    })
    
    for col_num, value in enumerate(text_df.columns.values):
        worksheet.write(0, col_num, value, header_format)
        
    # Set explicit column widths
    worksheet.set_column('A:A', 30, cell_format) 
    worksheet.set_column('B:C', 20, cell_format) 
    worksheet.set_column('D:D', 20, cell_format) 
    worksheet.set_column('E:E', 50, cell_format) 
    worksheet.set_column('F:F', 35, cell_format) 
    worksheet.set_column('G:G', 25, cell_format) 
    
    for idx, row in df.iterrows():
        # Lock Excel row height to ~146 pixels (110 points)
        worksheet.set_row(idx + 1, 110) 
        
        img = row.get('left_img_obj')
        if isinstance(img, Image.Image):
            # Calculate exactly to 135 pixels high so it leaves a 5px clean margin inside the cell
            target_height = 135
            aspect_ratio = img.width / img.height
            target_width = int(target_height * aspect_ratio)
            
            img_resized = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
            img_io = io.BytesIO()
            img_resized.save(img_io, format='PNG')
            
            # Insert the image perfectly centered with zero overlap
            worksheet.insert_image(
                idx + 1, 5, 
                f'img_{idx}.png', 
                {
                    'image_data': img_io, 
                    'x_offset': 5, 
                    'y_offset': 5,
                    'object_position': 1 
                }
            )
            
    writer.close()
    return output.getvalue()

# ----------------------------------------------------------------------
# User Interface & Workflow Routing
# ----------------------------------------------------------------------
st.title("🎬 SyncSketch PDF to Excel Converter")

workflow_mode = st.radio(
    "Select Workflow Mode:",
    [
        "1. Single or Batch PDFs (Process separately & stack)", 
        "2. Combine Notes (Merge multiple PDFs by Timecode)"
    ],
    horizontal=True
)

st.write("---")

uploaded_pdfs = st.file_uploader("Upload SyncSketch PDFs", type=["pdf"], accept_multiple_files=True)

if st.button("Process PDFs") and uploaded_pdfs:
    if not gemini_key:
        st.error("🔑 Gemini API Key missing.")
    else:
        # Dynamic naming logic
        if len(uploaded_pdfs) == 1:
            st.session_state.default_filename = uploaded_pdfs[0].name.replace(".pdf", "").replace(".PDF", "")
        else:
            st.session_state.default_filename = "Combined_SyncSketch_Notes"
            
        with st.spinner("Processing Pages & Extracting Screenshots..."):
            raw_data = process_multiple_pdfs(uploaded_pdfs, gemini_key)
            df = pd.DataFrame(raw_data)
            
            if workflow_mode == "2. Combine Notes (Merge multiple PDFs by Timecode)":
                merged_df = df.groupby(['Scene/File Identification', 'Episode Timecode']).agg({
                    'Batch Timecode': 'first',
                    'Name': lambda x: ' & '.join(x.unique()),
                    'Note': lambda x: '\n\n---\n'.join([f"[{name}]: {note}" for name, note in zip(df.loc[x.index, 'Name'], x)]),
                    'left_img_obj': 'first',
                    'Source File': lambda x: ', '.join(x.unique())
                }).reset_index()
                st.session_state.final_df = merged_df
            else:
                st.session_state.final_df = df
                
            st.success("Extraction Complete!")

# ----------------------------------------------------------------------
# Final Preview & Excel Download
# ----------------------------------------------------------------------
if 'final_df' in st.session_state and not st.session_state.final_df.empty:
    st.write("---")
    st.subheader("📋 Step 2: Preview Final Dataset")
    
    display_df = st.session_state.final_df.drop(columns=['left_img_obj'], errors='ignore')
    st.dataframe(display_df, use_container_width=True)
    
    st.subheader("🚀 Step 3: Download Formatted Excel")
    
    custom_filename = st.text_input("Name your output file (optional):", value=st.session_state.get('default_filename', 'SyncSketch_Export'))
    
    clean_name = custom_filename.strip() if custom_filename.strip() else st.session_state.get('default_filename', 'SyncSketch_Export')
    if not clean_name.lower().endswith('.xlsx'):
        clean_name += '.xlsx'
        
    st.write("Click below to download your Master Excel file. The formatting, borders, and image boundaries have all been automatically optimized.")
    
    excel_data = generate_excel_with_images(st.session_state.final_df)
    
    st.download_button(
        label=f"📥 Download {clean_name}",
        data=excel_data,
        file_name=clean_name,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
