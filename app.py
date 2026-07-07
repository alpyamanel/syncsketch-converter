import streamlit as st
import fitz  # PyMuPDF
from PIL import Image
import io
import pandas as pd
import concurrent.futures
from google import genai
from google.genai import types
from pydantic import BaseModel

st.set_page_config(page_title="SyncSketch PDF to Excel", layout="wide")

MAX_WORKERS = 5 

# ----------------------------------------------------------------------
# Setup & Gemini Key
# ----------------------------------------------------------------------
gemini_key = None
if "GEMINI_API_KEY" in st.secrets:
    gemini_key = st.secrets["GEMINI_API_KEY"]
else:
    st.sidebar.title("Configuration")
    gemini_key = st.sidebar.text_input("Gemini API Key", type="password")

# ----------------------------------------------------------------------
# Core Logic: PDF Processing & Vision AI
# ----------------------------------------------------------------------
def process_pdf_rows(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    row_images = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        pix = page.get_pixmap(matrix=fitz.Matrix(3, 3))
        page_img = Image.open(io.BytesIO(pix.tobytes("png")))
        
        width, height = page_img.size
        row_height = height // 3 
        
        for i in range(3):
            top = i * row_height
            bottom = (i + 1) * row_height
            if top >= height:
                break
                
            row_img = page_img.crop((0, top, width, bottom))
            split_x = int(width * 0.55)
            left_img = row_img.crop((0, 0, split_x, row_img.height))
            right_img = row_img.crop((split_x, 0, width, row_img.height))
            
            row_images.append({"left": left_img, "right": right_img})
            
    return row_images

def extract_single_row(pair, api_key):
    left_img = pair['left']
    right_img = pair['right']
    client = genai.Client(api_key=api_key)
    
    left_bio = io.BytesIO()
    left_img.save(left_bio, format="PNG")
    right_bio = io.BytesIO()
    right_img.save(right_bio, format="PNG")
    
    prompt = """
    Analyze these two segments of a review row from a SyncSketch PDF.
    1. Extract 'Scene/File Identification' from top-right INSIDE the left image.
    2. Extract 'Batch Timecode' from bottom-centre near 'REC TC' in the left image.
    3. Extract 'Episode Timecode' from bottom-right in the left image.
    4. Extract reviewer's 'Name' (bold/black text before colon) from right image.
    5. Extract 'Note' (text after colon) from right image.
    Ignore green text. Mark missing/unclear strictly as [UNCLEAR].
    """
    
    class SyncSketchRow(BaseModel):
        scene_id: str
        batch_timecode: str
        episode_timecode: str
        reviewer_name: str
        reviewer_note: str

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[
                types.Part.from_bytes(data=left_bio.getvalue(), mime_type="image/png"),
                types.Part.from_bytes(data=right_bio.getvalue(), mime_type="image/png"),
                prompt
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=SyncSketchRow,
                temperature=0.0
            ),
        )
        import json
        return json.loads(response.text)
    except Exception as e:
        return {"scene_id": "[UNCLEAR]", "batch_timecode": "[UNCLEAR]", "episode_timecode": "[UNCLEAR]", "reviewer_name": "[UNCLEAR]", "reviewer_note": f"[ERROR: {str(e)}]"}

def generate_excel_with_images(df):
    """Creates an Excel file in memory and embeds the images into the cells."""
    output = io.BytesIO()
    # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    
    # Drop the image object column for the text part of the dataframe
    text_df = df.drop(columns=['left_img_obj'])
    text_df.to_excel(writer, sheet_name='SyncSketch Notes', index=False)
    
    workbook = writer.book
    worksheet = writer.sheets['SyncSketch Notes']
    
    # Formatting: Set column widths
    worksheet.set_column('A:E', 25)
    worksheet.set_column('F:F', 35) # Column F will hold the images
    worksheet.write('F1', 'Visual Capture') # Header for images
    
    # Formatting: Set row heights and embed images
    for idx, row in df.iterrows():
        worksheet.set_row(idx + 1, 90) # Height to fit the image
        img = row['left_img_obj']
        
        if isinstance(img, Image.Image):
            img_io = io.BytesIO()
            img.save(img_io, format='PNG')
            # Insert image. Scale it down to fit neatly in the cell
            worksheet.insert_image(idx + 1, 5, f'img_{idx}.png', 
                                   {'image_data': img_io, 'x_scale': 0.18, 'y_scale': 0.18, 'object_position': 1})
            
    writer.close()
    processed_data = output.getvalue()
    return processed_data

# ----------------------------------------------------------------------
# User Interface
# ----------------------------------------------------------------------
st.title("🎬 SyncSketch PDF to Excel Converter")
st.write("Extract notes and screenshots into a downloadable Excel file. You can then open this directly in Google Sheets!")

uploaded_pdfs = st.file_uploader("Upload SyncSketch PDFs", type=["pdf"], accept_multiple_files=True)

if st.button("Process PDFs") and uploaded_pdfs:
    if not gemini_key:
        st.error("🔑 Gemini API Key missing.")
    else:
        all_extracted_data = []
        with st.spinner("Extracting data and slicing images..."):
            for file in uploaded_pdfs:
                pdf_bytes = file.read()
                rows = process_pdf_rows(pdf_bytes)
                progress_bar = st.progress(0, text=f"Analyzing {file.name}...")
                
                extracted_results = [None] * len(rows)
                with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    futures = {executor.submit(extract_single_row, pair, gemini_key): i for i, pair in enumerate(rows)}
                    for count, future in enumerate(concurrent.futures.as_completed(futures)):
                        idx = futures[future]
                        extracted_results[idx] = future.result()
                        progress_bar.progress((count + 1) / len(rows))
                
                for index, parsed_vals in enumerate(extracted_results):
                    all_extracted_data.append({
                        "Scene ID": parsed_vals.get("scene_id", ""),
                        "Batch TC": parsed_vals.get("batch_timecode", ""),
                        "Episode TC": parsed_vals.get("episode_timecode", ""),
                        "Name": parsed_vals.get("reviewer_name", ""),
                        "Note": parsed_vals.get("reviewer_note", ""),
                        "left_img_obj": rows[index]['left']
                    })
                progress_bar.empty()
                
            st.session_state.final_df = pd.DataFrame(all_extracted_data)
            st.success("Extraction Complete!")

if 'final_df' in st.session_state and not st.session_state.final_df.empty:
    st.write("---")
    st.subheader("📋 Preview Dataset")
    display_df = st.session_state.final_df.drop(columns=['left_img_obj'], errors='ignore')
    st.dataframe(display_df, use_container_width=True)
    
    st.subheader("🚀 Download Ready")
    st.write("Click below to download your master Excel file. You can upload this directly into your Google Drive to open it in Google Sheets.")
    
    # Generate the Excel file in memory
    excel_data = generate_excel_with_images(st.session_state.final_df)
    
    st.download_button(
        label="📥 Download Excel File",
        data=excel_data,
        file_name="SyncSketch_Export.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
