from flask import Flask, render_template, request, redirect, url_for
from model_loader import load_models
from model_scripts.senti_analysis import sentiment_analysis
from model_scripts.personality_traits import traits_predict
from model_scripts.speech_summarizer import text_summarizer
from model_scripts.transcibe_link import generate_transcript
import pickle


app = Flask(__name__)
text = ""
models = load_models()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/save_data', methods=['GET'])
def save_data():
    youtube_link = request.args['youtube_link']
    print(youtube_link)
    # text = youtube_link

    #youtube transcript
    link_id = youtube_link.split('v=')[-1].split('&')[0]
    text, no_of_words = generate_transcript(link_id)

    #sentiment analysis
    sentiments = sentiment_analysis(text)
    print(sentiments)

    #personality traits
    per_traits = traits_predict(text, models)
    print(per_traits)

    #speech summarizer
    text_summ = text_summarizer(text)
    print(text_summ)

    # return redirect(url_for('home'))
    return render_template('index.html', sentiments = sentiments[0], sentiments_pos = round(sentiments[1]), sentiments_neg = round(sentiments[2]), per_traits = per_traits[0], text_summ = text_summ[0])


if __name__ == '__main__':
    app.run()

# app.run()
