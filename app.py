import streamlit as st
import fitz  # PyMuPDF
from PIL import Image
import io
import json
import os
import pandas as pd
import gspread
import concurrent.futures
from oauth2client.service_account import ServiceAccountCredentials
from google import genai
from google.genai import types
from pydantic import BaseModel

st.set_page_config(page_title="SyncSketch PDF to Google Sheets", layout="wide")

# ----------------------------------------------------------------------
# Multithreading Settings
# ----------------------------------------------------------------------
# Increase this if you have a paid Gemini tier. Keep at 5-10 for free tier to avoid rate limits.
MAX_WORKERS = 5 

# ----------------------------------------------------------------------
# Credentials Resolution
# ----------------------------------------------------------------------
gemini_key = None
creds_dict = None
target_folder_id = ""

if "GEMINI_API_KEY" in st.secrets and "g_credentials" in st.secrets:
    gemini_key = st.secrets["GEMINI_API_KEY"]
    creds_dict = dict(st.secrets["g_credentials"])
    if "TARGET_FOLDER_ID" in st.secrets:
        target_folder_id = st.secrets["TARGET_FOLDER_ID"]
    st.sidebar.success("🔒 System running securely via Cloud Secrets Manager.")
else:
    st.sidebar.title("Configuration (Local Fallback)")
    gemini_key = st.sidebar.text_input("Gemini API Key", type="password")
    g_creds_file = st.sidebar.file_uploader("Google Service Account JSON", type=["json"])
    target_folder_id = st.sidebar.text_input("Google Drive Folder ID (Optional)")
    if g_creds_file:
        try:
            creds_dict = json.load(g_creds_file)
        except Exception as e:
            st.sidebar.error(f"Invalid JSON file: {e}")

# ----------------------------------------------------------------------
# Core Logic: PDF Processing & Vision AI
# ----------------------------------------------------------------------
def process_pdf_rows(pdf_bytes):
    """Opens the PDF and splits each page horizontally into rows."""
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
    """Helper to process a single row for the ThreadPoolExecutor."""
    left_img = pair['left']
    right_img = pair['right']
    
    client = genai.Client(api_key=api_key)
    
    left_bio = io.BytesIO()
    left_img.save(left_bio, format="PNG")
    
    right_bio = io.BytesIO()
    right_img.save(right_bio, format="PNG")
    
    prompt = """
    Analyze these two segments of a review row from a SyncSketch PDF.
    
    From the LEFT Image (Screenshot containing embedded layout markers):
    1. Look closely at the top-right corner INSIDE the image frame to extract the 'Scene/File Identification'.
    2. Look closely at the bottom-centre of the image frame, specifically near 'REC TC' for the 'Batch Timecode'.
    3. Look closely at the bottom-right corner of the image frame for the 'Episode Timecode'.
    
    From the RIGHT Image (Text area):
    4. Find the bold or black text before the colon (:) which represents the reviewer's 'Name'.
    5. Find the text after the colon (:) which represents the 'Note'.
    
    CRITICAL INSTRUCTIONS:
    - Ignore any green text values (like FRAME, TIMECODE, OPEN ^).
    - If any value is missing, obscured, or unclear, mark it strictly as [UNCLEAR]. Do not attempt to guess.
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
        return json.loads(response.text)
    except Exception as e:
        return {
            "scene_id": "[UNCLEAR]",
            "batch_timecode": "[UNCLEAR]",
            "episode_timecode": "[UNCLEAR]",
            "reviewer_name": "[UNCLEAR]",
            "reviewer_note": f"[ERROR: {str(e)}]"
        }

def process_multiple_pdfs(uploaded_files, api_key):
    """Processes a list of PDFs using multithreading for speed."""
    all_extracted_data = []
    
    for file in uploaded_files:
        st.write(f"**Processing:** `{file.name}`")
        pdf_bytes = file.read()
        rows = process_pdf_rows(pdf_bytes)
        
        progress_bar = st.progress(0, text=f"Analyzing {len(rows)} rows with Gemini AI...")
        
        # Multithreaded Gemini Extraction
        extracted_results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Map retains the original order of the rows
            futures = [executor.submit(extract_single_row, pair, api_key) for pair in rows]
            
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                # Update progress bar as tasks complete (order doesn't matter for the bar)
                progress_bar.progress((i + 1) / len(rows))
            
            # Retrieve results in the exact original order
            for future in futures:
                extracted_results.append(future.result())
        
        for index, parsed_vals in enumerate(extracted_results):
            all_extracted_data.append({
                "Source File": file.name,
                "Scene/File Identification": parsed_vals.get("scene_id", "[UNCLEAR]"),
                "Batch Timecode": parsed_vals.get("batch_timecode", "[UNCLEAR]"),
                "Episode Timecode": parsed_vals.get("episode_timecode", "[UNCLEAR]"),
                "Name": parsed_vals.get("reviewer_name", "[UNCLEAR]"),
                "Note": parsed_vals.get("reviewer_note", "[UNCLEAR]"),
                "left_img_obj": rows[index]['left']
            })
            
        progress_bar.empty()
        
    return all_extracted_data

# ----------------------------------------------------------------------
# Google Workspace Integration (Multithreaded)
# ----------------------------------------------------------------------
def upload_single_image(drive_service, pil_img, filename, folder_id, row_index):
    """Helper to upload a single image to Drive, returning the URL and index."""
    if not isinstance(pil_img, Image.Image):
        return row_index, str(pil_img) if pd.notnull(pil_img) else ""
        
    bio = io.BytesIO()
    pil_img.save(bio, format="PNG")
    bio.seek(0)
    
    file_metadata = {'name': filename}
    if folder_id:
        file_metadata['parents'] = [folder_id]
        
    from googleapiclient.http import MediaIoBaseUpload
    media = MediaIoBaseUpload(bio, mimetype='image/png', resumable=True)
    
    uploaded_file = drive_service.files().create(
        body=file_metadata, media_body=media, fields='id, webViewLink'
    ).execute()
    
    drive_service.permissions().create(
        fileId=uploaded_file.get('id'), body={'type': 'anyone', 'role': 'reader'}
    ).execute()
    
    return row_index, uploaded_file.get('webViewLink')

def create_google_sheet(target_creds, sheet_title, dataframe, folder_id):
    scope = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_dict(target_creds, scope)
    gc = gspread.authorize(creds)
    
    from googleapiclient.discovery import build
    drive_service = build('drive', 'v3', credentials=creds)
    
    sh = gc.create(sheet_title)
    if folder_id:
        drive_service.files().update(
            fileId=sh.id, addParents=folder_id, removeParents='root', fields='id, parents'
        ).execute()
        
    worksheet = sh.get_worksheet(0)
    headers = ["Scene/File Identification", "Batch Timecode", "Episode Timecode", "Name", "Note", "Image Link", "Source File"]
    worksheet.append_row(headers)
    
    data_dicts = dataframe.to_dict('records')
    progress_bar = st.progress(0, text="Uploading visuals to Drive (Multithreaded)...")
    
    # Multithreaded Image Uploads
    image_urls = [None] * len(data_dicts)
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS * 2) as executor:
        futures = []
        for idx, row in enumerate(data_dicts):
            filename = f"{sheet_title}_row_{idx+1}.png"
            img_obj = row.get('left_img_obj') if 'left_img_obj' in row else row.get('Image Link')
            futures.append(executor.submit(upload_single_image, drive_service, img_obj, filename, folder_id, idx))
            
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            idx, url = future.result()
            image_urls[idx] = url
            progress_bar.progress((i + 1) / len(data_dicts))
            
    progress_bar.progress(1.0, text="Syncing cells to Google Sheets...")
    
    rows_to_write = []
    for idx, row in enumerate(data_dicts):
        rows_to_write.append([
            str(row.get('Scene/File Identification', '')),
            str(row.get('Batch Timecode', '')),
            str(row.get('Episode Timecode', '')),
            str(row.get('Name', '')),
            str(row.get('Note', '')),
            image_urls[idx],
            str(row.get('Source File', ''))
        ])
        
    worksheet.append_rows(rows_to_write)
    worksheet.format("A1:G1", {"textFormat": {"bold": True}})
    
    progress_bar.empty()
    return sh.url

# ----------------------------------------------------------------------
# User Interface & Workflow Routing
# ----------------------------------------------------------------------
st.title("🎬 SyncSketch Advanced Pipeline to Google Sheets")

workflow_mode = st.radio(
    "Select Workflow Mode:",
    [
        "1. Single or Batch PDFs (Process separately & stack)", 
        "2. Combine Notes (Merge multiple PDFs by Timecode)", 
        "3. Update Existing Excel (Append new PDF to Excel Sheet)"
    ],
    horizontal=True
)

st.write("---")

if workflow_mode == "1. Single or Batch PDFs (Process separately & stack)":
    uploaded_pdfs = st.file_uploader("Upload one or more SyncSketch PDFs", type=["pdf"], accept_multiple_files=True)
    if st.button("Process PDFs") and uploaded_pdfs:
        if not gemini_key:
            st.error("🔑 Gemini API Key missing.")
        else:
            with st.spinner("Extracting (Multithreaded)..."):
                raw_data = process_multiple_pdfs(uploaded_pdfs, gemini_key)
                st.session_state.final_df = pd.DataFrame(raw_data)
                st.success("Extraction Complete!")

elif workflow_mode == "2. Combine Notes (Merge multiple PDFs by Timecode)":
    st.info("Notes on the same Timecode and Scene ID from different PDFs will be merged into a single row.")
    uploaded_pdfs = st.file_uploader("Upload SyncSketch PDFs to combine", type=["pdf"], accept_multiple_files=True)
    
    if st.button("Process & Combine PDFs") and uploaded_pdfs:
        if not gemini_key:
            st.error("🔑 Gemini API Key missing.")
        else:
            with st.spinner("Extracting and Merging..."):
                raw_data = process_multiple_pdfs(uploaded_pdfs, gemini_key)
                df = pd.DataFrame(raw_data)
                
                def merge_notes(series):
                    return "\n\n".join([str(x) for x in series])
                
                merged_df = df.groupby(['Scene/File Identification', 'Episode Timecode']).agg({
                    'Batch Timecode': 'first',
                    'Name': lambda x: ' & '.join(x.unique()),
                    'Note': lambda x: '\n---\n'.join([f"[{name}]: {note}" for name, note in zip(df.loc[x.index, 'Name'], x)]),
                    'left_img_obj': 'first',
                    'Source File': lambda x: ', '.join(x.unique())
                }).reset_index()
                
                st.session_state.final_df = merged_df
                st.success("Extraction and Merge Complete!")

elif workflow_mode == "3. Update Existing Excel (Append new PDF to Excel Sheet)":
    col1, col2 = st.columns(2)
    with col1:
        uploaded_excel = st.file_uploader("1. Upload Existing Master Excel (.xlsx)", type=["xlsx"])
    with col2:
        uploaded_pdf_for_excel = st.file_uploader("2. Upload New SyncSketch PDF (.pdf)", type=["pdf"])
        
    if st.button("Append PDF to Excel") and uploaded_excel and uploaded_pdf_for_excel:
        if not gemini_key:
            st.error("🔑 Gemini API Key missing.")
        else:
            with st.spinner("Reading Excel and parsing new PDF..."):
                df_excel = pd.read_excel(uploaded_excel)
                raw_pdf_data = process_multiple_pdfs([uploaded_pdf_for_excel], gemini_key)
                df_pdf = pd.DataFrame(raw_pdf_data)
                combined_df = pd.concat([df_excel, df_pdf], ignore_index=True)
                st.session_state.final_df = combined_df
                st.success("Append Complete!")

# ----------------------------------------------------------------------
# Step 2: Final Preview & Google Sheet Export
# ----------------------------------------------------------------------
if 'final_df' in st.session_state and not st.session_state.final_df.empty:
    st.write("---")
    st.subheader("📋 Step 2: Preview Final Dataset")
    
    display_df = st.session_state.final_df.drop(columns=['left_img_obj'], errors='ignore')
    st.dataframe(display_df, use_container_width=True)
    
    st.subheader("🚀 Step 3: Push Dataset to Google Sheets")
    sheet_title_input = st.text_input("Google Sheet Name", value="SyncSketch_Master_Export")
    
    if st.button("Generate Final Google Sheet"):
        if not creds_dict:
            st.error("❌ Google Workspace Configuration Missing.")
        else:
            try:
                sheet_url = create_google_sheet(
                    target_creds=creds_dict,
                    sheet_title=sheet_title_input,
                    dataframe=st.session_state.final_df,
                    folder_id=target_folder_id
                )
                st.success("🎉 Target Document Created Successfully!")
                st.markdown(f"[🔗 Open Your Master Google Sheet Here]({sheet_url})")
            except Exception as ex:
                st.error(f"Failed to compile target data onto Google Cloud Assets: {ex}")
