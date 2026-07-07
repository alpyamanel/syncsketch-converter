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
# Credentials Resolution
# ----------------------------------------------------------------------
gemini_key = None
if "GEMINI_API_KEY" in st.secrets:
    gemini_key = st.secrets["GEMINI_API_KEY"]
    st.sidebar.success("🔒 System running securely via Cloud Secrets Manager.")
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

def process_multiple_pdfs(uploaded_files, api_key):
    all_extracted_data = []
    
    for file in uploaded_files:
        pdf_bytes = file.read()
        rows = process_pdf_rows(pdf_bytes)
        progress_bar = st.progress(0, text=f"Analyzing {file.name} with AI...")
        
        extracted_results = [None] * len(rows)
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(extract_single_row, pair, api_key): i for i, pair in enumerate(rows)}
            for count, future in enumerate(concurrent.futures.as_completed(futures)):
                idx = futures[future]
                extracted_results[idx] = future.result()
                progress_bar.progress((count + 1) / len(rows))
        
        for index, parsed_vals in enumerate(extracted_results):
            all_extracted_data.append({
                "Source File": file.name,
                "Scene/File Identification": parsed_vals.get("scene_id", ""),
                "Batch Timecode": parsed_vals.get("batch_timecode", ""),
                "Episode Timecode": parsed_vals.get("episode_timecode", ""),
                "Name": parsed_vals.get("reviewer_name", ""),
                "Note": parsed_vals.get("reviewer_note", ""),
                "left_img_obj": rows[index]['left']
            })
        progress_bar.empty()
        
    return all_extracted_data

# ----------------------------------------------------------------------
# Excel Generation with Advanced Formatting
# ----------------------------------------------------------------------
def generate_excel_with_images(df):
    """Creates a beautifully formatted Excel file in memory with embedded images."""
    output = io.BytesIO()
    
    # Isolate the text data and prep an empty column for our images
    text_df = df.drop(columns=['left_img_obj'], errors='ignore')
    text_df.insert(5, 'Visual Capture', '') 
    
    # Initialize the Excel Writer using the xlsxwriter engine
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    text_df.to_excel(writer, sheet_name='SyncSketch Notes', index=False)
    
    workbook = writer.book
    worksheet = writer.sheets['SyncSketch Notes']
    
    # Create beautiful formatting profiles
    header_format = workbook.add_format({
        'bold': True,
        'valign': 'vcenter',
        'align': 'center',
        'bg_color': '#D3D3D3', # Light Gray Background
        'border': 1
    })
    
    cell_format = workbook.add_format({
        'text_wrap': True,    # CRITICAL: Forces text to stay inside the cell
        'valign': 'vcenter',  # Centers text vertically next to the image
        'border': 1           # Adds clean grid lines
    })
    
    # Overwrite headers with our custom header format
    for col_num, value in enumerate(text_df.columns.values):
        worksheet.write(0, col_num, value, header_format)
        
    # Apply column widths and the text wrapping format
    worksheet.set_column('A:A', 30, cell_format) # Scene ID
    worksheet.set_column('B:C', 20, cell_format) # Timecodes
    worksheet.set_column('D:D', 20, cell_format) # Name
    worksheet.set_column('E:E', 50, cell_format) # Note (Extra wide for lots of text)
    worksheet.set_column('F:F', 35, cell_format) # Visual Capture (Image column)
    worksheet.set_column('G:G', 25, cell_format) # Source File
    
    # Embed the images and set explicit row heights
    for idx, row in df.iterrows():
        # Set row height to 110 pixels to comfortably fit the image
        worksheet.set_row(idx + 1, 110) 
        
        img = row.get('left_img_obj')
        if isinstance(img, Image.Image):
            img_io = io.BytesIO()
            img.save(img_io, format='PNG')
            
            # Insert the image into Column F (Index 5).
            # x_offset and y_offset give it a 5px margin so it doesn't touch the borders.
            # object_position: 2 ensures the image moves with the cells but maintains aspect ratio.
            worksheet.insert_image(
                idx + 1, 5, 
                f'img_{idx}.png', 
                {
                    'image_data': img_io, 
                    'x_scale': 0.22, 
                    'y_scale': 0.22, 
                    'x_offset': 5, 
                    'y_offset': 5,
                    'object_position': 2 
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
        with st.spinner("Extracting (Multithreaded)..."):
            raw_data = process_multiple_pdfs(uploaded_pdfs, gemini_key)
            df = pd.DataFrame(raw_data)
            
            if workflow_mode == "2. Combine Notes (Merge multiple PDFs by Timecode)":
                # Combine matching timecodes and merge the notes text
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
    
    # Hide the raw PIL image object from the browser preview
    display_df = st.session_state.final_df.drop(columns=['left_img_obj'], errors='ignore')
    st.dataframe(display_df, use_container_width=True)
    
    st.subheader("🚀 Step 3: Download Formatted Excel")
    st.write("Click below to download your Master Excel file. The formatting, borders, and image boundaries have all been automatically optimized.")
    
    # Generate the formatted Excel file in memory
    excel_data = generate_excel_with_images(st.session_state.final_df)
    
    st.download_button(
        label="📥 Download Master Excel File",
        data=excel_data,
        file_name="SyncSketch_Export.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
