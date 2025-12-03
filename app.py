"""
Streamlit app to run Kraken OCR with the Vilne Yiddish model (corrected version).



Usage:
  1. Install dependencies: pip install streamlit pillow kraken
  2. Run: streamlit run app.py

This is a corrected copy of `app.py` with safer string handling in the sidebar.
"""

import os
import tempfile
import subprocess
import streamlit as st
from PIL import Image

DEFAULT_MODEL_PATH = "models/vilne_yiddish_8a.mlmodel"

st.set_page_config(page_title="My Kraken Model", layout="centered")
st.title("My Kraken Model")
st.write("Upload an image; text will be recognized using the Kraken HTR model.")

model_path = st.text_input("Model path", DEFAULT_MODEL_PATH)

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "tif", "tiff"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("L")
    except Exception:
        # try open as RGB
        image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded image", use_container_width=True)

    if st.button("Run OCR"):
        tmpdir = tempfile.mkdtemp()
        input_path = os.path.join(tmpdir, "input.png")
        output_path = os.path.join(tmpdir, "output.txt")
        image.save(input_path)

        # Try Kraken CLI first
        cli_cmd = [
            "kraken",
            "-i",
            input_path,
            output_path,
            "binarize",
            "segment",
            "ocr",
            "-m",
            model_path,
        ]

        st.text("Attempting to run Kraken CLI...")
        try:
            proc = subprocess.run(cli_cmd, capture_output=True, text=True, check=True)
            # Read the output file and show recognized text
            if os.path.exists(output_path):
                with open(output_path, "r", encoding="utf8") as fh:
                    result = fh.read()
                st.success("OCR completed via Kraken CLI")
                st.text_area("Recognized text", value=result, height=300)
            else:
                st.error("Kraken CLI ran but no output file created.")
                st.write(proc.stdout)
                st.write(proc.stderr)

        except FileNotFoundError:
            st.warning("Kraken CLI not found — falling back to Python API (if available).\nInstall CLI with: `pip install kraken` or make sure `kraken` is in your PATH.")
            # Try Python API fallback
            try:
                """
                Streamlit OCR app (professional-grade) for Yiddish using Kraken.

                Features:
                - Advanced pre-processing: configurable binarization threshold and processed image preview.
                - Smart workflow: send processed image to Kraken CLI (binarize, segment, ocr sequence).
                - RTL support: output area styled for right-to-left languages.
                - Download recognized text as a .txt file.

                Usage:
                  pip install -r requirements.txt
                  streamlit run app.py
                """

                import os
                import tempfile
                import subprocess
                import streamlit as st
                from PIL import Image

                DEFAULT_MODEL_PATH = "models/vilne_yiddish_8a.mlmodel"


                st.set_page_config(page_title="Vilne Yiddish OCR", layout="centered")
                st.title("Vilne Yiddish OCR — Kraken")
                st.write("Upload an image and tune the binarization threshold until the letters are crisp.")


                # Sidebar controls
                st.sidebar.header("Settings")
                model_path = st.sidebar.text_input("Model path", DEFAULT_MODEL_PATH)
                threshold = st.sidebar.slider("Binarization Threshold", min_value=0, max_value=255, value=128)

                st.sidebar.markdown("---")
                st.sidebar.markdown("Dependencies:")
                st.sidebar.code("pip install -r requirements.txt")


                def apply_threshold(image: Image.Image, thresh: int) -> Image.Image:
                    """Convert to grayscale and apply a manual threshold returning a black/white image."""
                    gray = image.convert("L")
                    # Use point to threshold: pixels > thresh -> 255 (white), else 0 (black)
                    bw = gray.point(lambda p: 255 if p > thresh else 0)
                    return bw


                uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "tif", "tiff"])

                result_text = None

                if uploaded_file is not None:
                    # Open original image
                    try:
                        orig = Image.open(uploaded_file)
                    except Exception as e:
                        st.error(f"Failed to open image: {e}")
                        orig = None

                    if orig is not None:
                        st.subheader("Original Image")
                        st.image(orig, use_container_width=True)

                        # Apply threshold preview
                        processed = apply_threshold(orig, threshold)
                        st.subheader("Processed Image (Black & White)")
                        st.image(processed, caption=f"Threshold = {threshold}", use_container_width=True)

                        # Begin OCR when user clicks
                        if st.button("Run OCR"):
                            tmpdir = tempfile.mkdtemp()
                            processed_input_path = os.path.join(tmpdir, "input_processed.png")
                            output_path = os.path.join(tmpdir, "output.txt")

                            # Save processed image to a file (PNG preserves binary values)
                            processed.save(processed_input_path)

                            # Kraken CLI: pass the processed image so Kraken sees a clean image.
                            cli_cmd = [
                                "kraken",
                                "-i",
                                processed_input_path,
                                output_path,
                                "binarize",
                                "segment",
                                "ocr",
                                "-m",
                                model_path,
                            ]

                            st.text("Attempting to run Kraken CLI...")
                            try:
                                proc = subprocess.run(cli_cmd, capture_output=True, text=True, check=True)

                                if os.path.exists(output_path):
                                    with open(output_path, "r", encoding="utf8") as fh:
                                        result_text = fh.read()
                                    st.success("OCR completed via Kraken CLI")
                                else:
                                    st.error("Kraken CLI ran but did not produce an output file.")
                                    st.write(proc.stdout)
                                    st.write(proc.stderr)

                            except FileNotFoundError:
                                st.warning("Kraken CLI not found — attempting Kraken Python API fallback.")
                                try:
                                    from kraken import rpred, serialization

                                    st.text("Loading model and running Kraken Python API...")
                                    model = serialization.load_any_model(model_path)

                                    # Try recognize on the processed image
                                    try:
                                        preds = rpred.recognize(model, [processed])
                                    except Exception:
                                        preds = rpred.rpred(model, [processed])

                                    if isinstance(preds, list):
                                        result_text = "\n".join([str(p) for p in preds])
                                    else:
                                        result_text = str(preds)

                                    st.success("OCR completed via Kraken Python API")

                                except Exception as e:
                                    st.error(f"Failed to use Kraken Python API: {e}")
                                    st.info("Install the 'kraken' package or ensure the 'kraken' CLI is available in PATH.")

                            except subprocess.CalledProcessError as e:
                                st.error(f"Kraken CLI failed: {e}")
                                st.write("STDOUT:\n", e.stdout)
                                st.write("STDERR:\n", e.stderr)

                            except Exception as e:
                                st.error(f"Unexpected error while running OCR: {e}")

                        # end Run OCR button

                        # Inject CSS for RTL on the Recognized text area (targets aria-label)
                        st.markdown(
                            """
                            <style>
                            textarea[aria-label="Recognized text"] {
                                direction: rtl;
                                text-align: right;
                                font-family: 'DejaVu Sans', sans-serif;
                                white-space: pre-wrap;
                            }
                            </style>
                            """,
                            unsafe_allow_html=True,
                        )

                        # If result_text is available, show and allow download
                        if result_text:
                            st.subheader("Recognized text")
                            st.text_area("Recognized text", value=result_text, height=300)

                            # Add download button
                            st.download_button(
                                label="Download Text",
                                data=result_text,
                                file_name="recognized.txt",
                                mime="text/plain",
                            )


                # End of app
