import streamlit as st
import joblib
import yake
import os
from transformers import pipeline
from newsapi import NewsApiClient

api_key = os.getenv("API_KEY")

newsapi = NewsApiClient(api_key=api_key)

st.set_page_config(page_title="TextSumm",layout="wide")

page_bg="""
    <style>
    [data-testid='stAppViewContainer']{
    background-image: url("https://towardsdatascience.com/a-quick-introduction-to-text-summarization-in-machine-learning-3d27ccf18a9f");
    background-size: cover;
    }
    [data-testid='stHeader']{
    background-image: url("https://towardsdatascience.com/a-quick-introduction-to-text-summarization-in-machine-learning-3d27ccf18a9f");
    background-size: cover;
    }
    </style>
"""
vectoriser=joblib.load('Vectorizer')
Classifier=joblib.load('NaiveBayes')
st.title('Text Summarizer')
st.text("Have patience,the processing may take some time")
col1,col2= st.columns(2)

with col1:
    form = st.form(key='my_form')
    text=form.text_input(label='Enter Text')
    form.text_area("You Entered",text)
    # text = st.text_input( "Enter some text 👇")
    # button=st.button("Summarize")
    # st.write("\n")
    length = form.slider('Set Maximum Length of the Summary', 50,150,250, step=100)
    button = form.form_submit_button(label='Summarize')


summariser=pipeline('summarization')
summary_response=""
if button:
    # with st.spinner('Processing...'):
    #     time.sleep(30)
    with col2:
        if len(text)!=0:
            summary_response=summariser(text,max_length=length,min_length=length//2,do_sample=False)[0]['summary_text']
            st.text_area("Summarized text",summary_response)
            st.text("Length of Summary")
            st.text(len(summary_response.split(" ")))
    # button=True
    with st.sidebar:
        st.header("Keywords")
        keyword_extractor = yake.KeywordExtractor()
        keywords = keyword_extractor.extract_keywords(summary_response)
        keyword_list = [kw[0] for kw in keywords]
        st.write(keyword_list)
    # button=True
    with col2:
        result=Classifier.predict(vectoriser.transform([summary_response]))
        st.markdown("### Predicted Article Category : ")
        st.text(result[0])

    top_headlines=newsapi.get_top_headlines(q=result[0],
                                              language='en',
                                              country='in')

    # total=top_headlines['totalResults']
    # st.text(total)
    st.markdown("### Articles you may be interested in...")
    # for i in range(0,total):
    imagecol,textcol=st.columns((1,2))
    with imagecol:
        image = top_headlines['articles'][0]['urlToImage']
        st.image(image)
    with textcol:
        title=top_headlines['articles'][0]['title']
        link=top_headlines['articles'][0]['url']
        st.markdown(title)
        st.link_button("Visit",link)





