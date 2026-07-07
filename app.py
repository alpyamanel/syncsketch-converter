import streamlit as st
import fitz  # PyMuPDF
from PIL import Image
import io
import json
import os
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from google import genai
from google.genai import types

st.set_page_config(page_title="SyncSketch PDF to Google Sheets", layout="wide")

# ----------------------------------------------------------------------
# Credentials Resolution (Production Secrets vs Local Sidebar Fallback)
# ----------------------------------------------------------------------
gemini_key = None
creds_dict = None
target_folder_id = ""

# 1. Check for Production Cloud Secrets
if "GEMINI_API_KEY" in st.secrets and "g_credentials" in st.secrets:
    gemini_key = st.secrets["GEMINI_API_KEY"]
    # Convert the secrets mapping directly into a standard dict for gspread
    creds_dict = dict(st.secrets["g_credentials"])
    if "TARGET_FOLDER_ID" in st.secrets:
        target_folder_id = st.secrets["TARGET_FOLDER_ID"]
    
    st.sidebar.success("🔒 System running securely via Cloud Secrets Manager.")
else:
    # 2. Fallback to Sidebar UI for Local Testing/Manual Uploads
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
# Session State Initialization
# ----------------------------------------------------------------------
if 'extracted_data' not in st.session_state:
    st.session_state.extracted_data = None
if 'processed_pdf_name' not in st.session_state:
    st.session_state.processed_pdf_name = ""

# ----------------------------------------------------------------------
# Core Logic: PDF Processing & Image Cropping
# ----------------------------------------------------------------------
def process_pdf_rows(pdf_bytes):
    """Opens the PDF and splits each page horizontally into rows."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    row_images = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        # Render page to high-res image (300 DPI for better vision processing)
        pix = page.get_pixmap(matrix=fitz.Matrix(3, 3))
        page_img = Image.open(io.BytesIO(pix.tobytes("png")))
        
        width, height = page_img.size
        row_height = height // 3 
        
        for i in range(3):
            top = i * row_height
            bottom = (i + 1) * row_height
            if top >= height:
                break
                
            # Crop the entire bounding row
            row_img = page_img.crop((0, top, width, bottom))
            
            # Split into Left Image (Screenshot) and Right Column (Notes)
            split_x = int(width * 0.55)
            left_img = row_img.crop((0, 0, split_x, row_img.height))
            right_img = row_img.crop((split_x, 0, width, row_img.height))
            
            row_images.append({
                "left": left_img,
                "right": right_img
            })
            
    return row_images

# ----------------------------------------------------------------------
# Gemini Vision API Analysis
# ----------------------------------------------------------------------
def extract_data_with_gemini(left_img, right_img, api_key):
    """Uses Gemini 2.5 Flash via Structured Outputs to analyze row data."""
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
    
    class SyncSketchRow(types.BaseModel):
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

# ----------------------------------------------------------------------
# Google Workspace Integration
# ----------------------------------------------------------------------
def upload_image_to_drive(drive_service, pil_img, filename, folder_id):
    """Uploads a cropped PNG to Google Drive and returns its viewable link."""
    bio = io.BytesIO()
    pil_img.save(bio, format="PNG")
    bio.seek(0)
    
    file_metadata = {'name': filename}
    if folder_id:
        file_metadata['parents'] = [folder_id]
        
    from googleapiclient.http import MediaIoBaseUpload
    media = MediaIoBaseUpload(bio, mimetype='image/png', resumable=True)
    
    uploaded_file = drive_service.files().create(
        body=file_metadata,
        media_body=media,
        fields='id, webViewLink'
    ).execute()
    
    # Adjust permissions so anyone with the link can view the thumbnail inside Sheets
    drive_service.permissions().create(
        fileId=uploaded_file.get('id'),
        body={'type': 'anyone', 'role': 'reader'}
    ).execute()
    
    return uploaded_file.get('webViewLink')

def create_google_sheet(target_creds, sheet_title, data_rows, folder_id):
    """Creates a formatted Google Sheet and maps data rows into it."""
    scope = [
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/drive'
    ]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(target_creds, scope)
    gc = gspread.authorize(creds)
    
    from googleapiclient.discovery import build
    drive_service = build('drive', 'v3', credentials=creds)
    
    sh = gc.create(sheet_title)
    
    if folder_id:
        drive_service.files().update(
            fileId=sh.id,
            addParents=folder_id,
            removeParents='root',
            fields='id, parents'
        ).execute()
        
    worksheet = sh.get_worksheet(0)
    
    headers = [
        "Scene/File Identification", 
        "Batch Timecode", 
        "Episode Timecode", 
        "Name", 
        "Note", 
        "Image Link"
    ]
    worksheet.append_row(headers)
    
    progress_bar = st.progress(0, text="Uploading visual captures to Drive & syncing cells...")
    
    rows_to_write = []
    for idx, row in enumerate(data_rows):
        filename = f"{sheet_title}_row_{idx+1}.png"
        img_url = upload_image_to_drive(drive_service, row['left_img_obj'], filename, folder_id)
        
        rows_to_write.append([
            row['Scene/File Identification'],
            row['Batch Timecode'],
            row['Episode Timecode'],
            row['Name'],
            row['Note'],
            img_url
        ])
        progress_bar.progress((idx + 1) / len(data_rows))
        
    worksheet.append_rows(rows_to_write)
    worksheet.format("A1:F1", {"textFormat": {"bold": True}})
    
    return sh.url

# ----------------------------------------------------------------------
# User Interface (Streamlit)
# ----------------------------------------------------------------------
st.title("🎬 SyncSketch PDF Pipeline to Google Sheets")
st.write("Convert visual SyncSketch breakdown printouts into cleanly mapped structured datasets.")

uploaded_file = st.file_uploader("Upload SyncSketch Review Notes PDF", type=["pdf"])

if uploaded_file is not None:
    if not gemini_key:
        st.error("🔑 Configuration Missing: Please set your Gemini API Key in the cloud secrets dashboard or sidebar.")
    else:
        if st.session_state.processed_pdf_name != uploaded_file.name:
            with st.spinner("Slicing PDF layouts into processing matrices..."):
                pdf_bytes = uploaded_file.read()
                rows = process_pdf_rows(pdf_bytes)
                
                extracted_dataset = []
                status_text = st.empty()
                
                for index, pair in enumerate(rows):
                    status_text.text(f"Analyzing Row Element {index+1} of {len(rows)} via Gemini AI...")
                    parsed_vals = extract_data_with_gemini(pair['left'], pair['right'], gemini_key)
                    
                    extracted_dataset.append({
                        "Scene/File Identification": parsed_vals.get("scene_id", "[UNCLEAR]"),
                        "Batch Timecode": parsed_vals.get("batch_timecode", "[UNCLEAR]"),
                        "Episode Timecode": parsed_vals.get("episode_timecode", "[UNCLEAR]"),
                        "Name": parsed_vals.get("reviewer_name", "[UNCLEAR]"),
                        "Note": parsed_vals.get("reviewer_note", "[UNCLEAR]"),
                        "left_img_obj": pair['left']
                    })
                    
                status_text.empty()
                st.session_state.extracted_data = extracted_dataset
                st.session_state.processed_pdf_name = uploaded_file.name

        if st.session_state.extracted_data:
            st.subheader("📋 Step 1: Preview Parsed Extraction Content")
            
            preview_list = []
            for row in st.session_state.extracted_data:
                preview_list.append({
                    "Scene ID": row["Scene/File Identification"],
                    "Batch TC": row["Batch Timecode"],
                    "Episode TC": row["Episode Timecode"],
                    "Reviewer": row["Name"],
                    "Note": row["Note"]
                })
            st.dataframe(preview_list, use_container_width=True)
            
            st.write("---")
            st.subheader("🚀 Step 2: Push Dataset to Cloud Workspaces")
            
            sheet_title_input = st.text_input("Google Sheet Name", value=f"SyncSketch_Export_{uploaded_file.name.replace('.pdf', '')}")
            
            if st.button("Generate Google Sheet"):
                if not creds_dict:
                    st.error("❌ Google Workspace Configuration Missing: Please supply your service account details.")
                else:
                    try:
                        with st.spinner("Provisioning cloud spreadsheet system..."):
                            sheet_url = create_google_sheet(
                                target_creds=creds_dict,
                                sheet_title=sheet_title_input,
                                data_rows=st.session_state.extracted_data,
                                folder_id=target_folder_id
                            )
                            st.success("🎉 Target Document Created Successfully!")
                            st.markdown(f"[🔗 Open Google Sheet Here]({sheet_url})")
                    except Exception as ex:
                        st.error(f"Failed to compile target data onto Google Cloud Assets: {ex}")
