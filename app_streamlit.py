"""
Streamlit app to run Kraken OCR with the Vilne Yiddish model (corrected version).

Usage:
  1. Install dependencies: pip install streamlit pillow kraken
  2. Run: streamlit run app_streamlit.py

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

    st.image(image, caption="Uploaded image", use_column_width=True)

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
            "ocr",
            "-m",
            model_path,
            "--threads",
            "1",
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
                from kraken import rpred, serialization
                st.text("Running Kraken using Python API...")
                model = serialization.load_any_model(model_path)

                # Kraken's rpred API expects segmented regions/line images. We try a simple
                # direct prediction on the full image (if supported) and otherwise attempt
                # a best-effort segmentation by rows.
                try:
                    preds = rpred.recognize(model, [image])
                except Exception:
                    # Try rpred.rpred as another option: rpred.rpred(model, images)
                    preds = rpred.rpred(model, [image])

                # Preds may be a list of strings or objects; try to handle both
                if isinstance(preds, list):
                    text_out = "\n".join([str(p) for p in preds])
                else:
                    text_out = str(preds)

                st.success("OCR completed via Kraken Python API")
                st.text_area("Recognized text", value=text_out, height=300)

            except Exception as e:
                st.error("Failed to use Kraken Python API: {}".format(e))
                st.info("Make sure the 'kraken' package is installed (pip install kraken) or the 'kraken' CLI is available in PATH")

        except subprocess.CalledProcessError as e:
            st.error("Kraken CLI failed: {}".format(e))
            st.write("STDOUT:\n", e.stdout)
            st.write("STDERR:\n", e.stderr)

        except Exception as e:
            st.error("Unexpected error while running OCR: {}".format(e))

# Show a small help section for dependencies
st.sidebar.header("Run instructions")
st.sidebar.markdown("1. Install the required dependencies in your environment:")
st.sidebar.code("pip install streamlit pillow kraken")
st.sidebar.markdown("2. Run the app:")
st.sidebar.code("streamlit run app_streamlit.py")

st.sidebar.markdown("If you prefer the Kraken CLI (recommended), ensure `kraken` is on your PATH.")
