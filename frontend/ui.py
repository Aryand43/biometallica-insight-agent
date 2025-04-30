import streamlit as st
import os
import time
import uuid
from backend.pdf_parser import extract_text_from_pdf
from backend.llm_agent import generate_insight_report

def main_ui():
    st.title("BioMetallica Insight Agent")
    st.markdown("Turn scanned proposals into strategic reports using AI.")

    os.makedirs("data/uploads", exist_ok=True)

    st.header("Upload Proposal")
    uploaded_file = st.file_uploader("Upload a PDF (scanned or digital)", type=["pdf"])

    if uploaded_file:
        unique_filename = f"{uuid.uuid4()}_{uploaded_file.name}"
        file_path = f"data/uploads/{unique_filename}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

        with st.spinner("Extracting text from PDF..."):
            try:
                extracted_text = extract_text_from_pdf(file_path)
                st.success("Text extracted successfully.")

                verbose = st.checkbox("Enable verbose logging")
                if verbose:
                    st.subheader("Debug: First 2000 characters of extracted text")
                    st.code(extracted_text[:2000])

                st.markdown("---")
                st.header("Preview Extracted Text")
                with st.expander("View Extracted Text", expanded=False):
                    st.write(extracted_text[:8000])  

                st.markdown("---")
                st.header("Generate AI Insight Report")
                custom_filename = st.text_input("Optional: Rename download file", value="insight_report.txt")

                if st.button("Generate Report"):
                    st.markdown("Generating report... Please wait.") 

                    start_time = time.time()
                    with st.spinner("Running DeepSeek..."):
                        try:
                            report = generate_insight_report(extracted_text)
                            duration = time.time() - start_time
                            st.success(f"Report generated in {duration:.2f} seconds.")

                            st.subheader("AI-Generated Insight Report")
                            with st.expander("View Full Report", expanded=True):
                                st.write(report)

                            st.download_button(
                                label="Download Report",
                                data=report,
                                file_name=custom_filename if custom_filename else "insight_report.txt",
                                mime="text/plain"
                            )

                        except Exception as e:
                            st.error(f"Insight generation failed: {e}")

            except Exception as e:
                st.error(f"Text extraction failed: {e}")
