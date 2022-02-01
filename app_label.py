from st_aggrid import AgGrid
import streamlit as st
from PIL import Image
import cv2
import numpy as np
import pandas as pd



@st.cache  # 👈 Added this
def clean_img(img):
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 21)
    height, width = img.shape[:2]
    img = Image.fromarray(img).convert('RGB')
    newsize = (int(width / 3), int(height / 3))
    img = img.resize(newsize)
    return img


def get_words_pos(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    thresh = cv2.threshold(blur, 125, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    # Create rectangular structuring element and dilate
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 3))
    dilate = cv2.dilate(thresh, kernel, iterations=7)
    df = pd.DataFrame(columns=['Bottom', 'Left', 'Right', 'Top'])
    count = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    count = count[0] if len(count) == 2 else count[1]
    i = len(count)-1
    for c in count:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(image, (x, y), (x + w, y + h), (60, 0, 0), 2)  # change to convenient number width
        cv2.putText(image, str(i), (x - 2, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (223, 93, 25), 2)
        df.loc[i] = (y + w, x + 10, x + h, y + 10)
        ROI = image[y:y + h, x:x + w]
        cv2.imwrite("imgs/" + str(i) + '.png', ROI)
        i -= 1
    return image, df


def image_input() -> object:
    st.sidebar.markdown("<h4 style='text-align: center; '>Please don't put files with the same name"
                        "</h4>", unsafe_allow_html=True)

    content_file = st.sidebar.file_uploader("", type=["png", "jpg", "jpeg"], key='first')

    if content_file is not None:
        content_gen = Image.open(content_file)
        img = clean_img(content_gen)
        img.save("images/" + str(content_file.name))
        content = np.array(img)
        content = cv2.cvtColor(content, cv2.COLOR_RGB2BGR)
        image, positions = get_words_pos(content)
        positions.to_csv("Positions/" + str(content_file.name) + ".csv")
    else:
        st.warning("Waiting for you to upload your image")
        st.stop()
    return image, content_file.name





st.set_page_config(layout="wide")
st.markdown("<h2 style='text-align: center; '>Annotation of Text in Images</h2>", unsafe_allow_html=True)

content, name = image_input()  # get image from input and file name
df = pd.read_csv("Positions/" + str(name) + ".csv")

words=pd.DataFrame(np.empty((len(df.index), 1), dtype=object), columns=["Words_from_image"])
# Print Image
lst = list(np.arange(0,len(df)))
words["number of image"]=lst
# get image from input and file name
for i in range(0,len(df)):
    img = Image.open("imgs/"+str(i)+".png")
    image = img.resize((1600, 900), Image.LANCZOS)
    st.sidebar.image(image, width=100,  caption=str(i))
# Changing columns name with index number
#df = df.rename(columns={df.columns[0]: 'number on image'})
grid_return = AgGrid(words, editable=True, fit_columns_on_grid_load = True,reload_data=False)
new_df = grid_return['data']

df = df.loc[::-1].reset_index(drop=True)
new_df[['Bottom', 'Left', 'Right', 'Top']] = df[['Bottom', 'Left', 'Right', 'Top']]
st.download_button(
    label="Download the annotation",
    data=new_df.to_csv(),
    file_name=str(name) + '.csv'
)

st.image(content,use_column_width=False)
#new_df.to_csv("labels_app/" + str(name) + ".csv", encoding="ISO-8859-1")
