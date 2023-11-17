import streamlit as st
from streamlit_card import card

    
def main():
    st.title("Choose the right Option 🚀")
    col1, col2 = st.columns(2)
    with col1:
        card(title="PDFGPT", text="Submit PDF. Get QNA.", key="c1", image="https://e0.pxfuel.com/wallpapers/1007/23/desktop-wallpaper-acrobat-29-adobe.jpg", url="http://127.0.0.1:8501")
    with col2:
        card(title="XIVGPT", text="Submit ARXIV DOI. Get QNA.", key="c2", image="https://i.imgur.com/7wYWcy7l.jpg", url="http://127.0.0.1:8502")
    col3, col4 = st.columns(2)
    with col3:
        card(title="IONGPT", text="Submit image with texts. Get QNA.", key="c3", image="https://c4.wallpaperflare.com/wallpaper/157/295/672/books-business-college-copy-wallpaper-preview.jpg", url="http://127.0.0.1:8502")
    with col4:
        card(title="WEBGPT", text="Submit WikiPedia link. Get QNA.", key="c4", image="https://upload.wikimedia.org/wikipedia/commons/thumb/7/77/10_sharing_book_cover_background.jpg/691px-10_sharing_book_cover_background.jpg", url="http://127.0.0.1:8502")

if __name__ == "__main__":
    main()