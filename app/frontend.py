import requests
import streamlit as st



# ----------------- Header ---------------
st.title("Water Bodies Segmentation")



# --------------- File uploading ------------------
uploading = st.container()
uploaded_file = uploading.file_uploader(
    label="Upload an image or a video",
    type=['png', 'jpg', 'jpeg'],
    accept_multiple_files=False
)



# ------------- Submit ---------------
submit = None
if uploaded_file:
    submitting = st.container()
    submit = submitting.button("Submit")
    st.markdown("""---""")


# -------------- uploaded file ------------
generating = st.container()
if submit:

    # Save uploaded image
    with open(f"saved_images/{uploaded_file.name}", "wb") as file:
        file.write(uploaded_file.getbuffer())
    
    # Make a request with image path
    data = {
        "image_name": uploaded_file.name
    }
        
    response = requests.post(
        "http://127.0.0.1:8000/segment",
        json=data
    )


    # Show images
    images = st.container()
    images_columns = images.columns([2, 1])
    images_columns[0].image(uploaded_file)