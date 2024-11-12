import flask
from flask_cors import CORS
from flask import render_template, url_for, redirect, flash,request
import torch
from summarizer import Summarizer


from spacy_summarization import text_summarizer
#from gensim.summarization import summarize
from nltk_summarization import nltk_summarizer
import time
import spacy
import assemblyai as aai
#nlp = spacy.load('en')
nlp = spacy.load("en_core_web_sm")



app = flask.Flask(__name__)

CORS(app)

# Set the API key for AssemblyAI
aai.settings.api_key = "3e65b9ef5e3b4d8fa87157511b012be1"

# Function to transcribe audio file and return text
def transcribe_audio(audio_file_path):
    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(audio_file_path)
    transcript_text = transcript.text
    return transcript_text


model = Summarizer()
@app.route("/", methods=["POST", "GET"])
def index():
	return render_template("index.html")	

@app.route("/summarize", methods=["POST", "GET"])
def summarize():
	data = dict()
	data["success"] = False

	if flask.request.method == "POST":
    # Check if an audio file is uploaded
		if "audio_file" not in flask.request.files:
			#flash("No audio file uploaded", "warning")
			return render_template("index.html")
		audio_file = flask.request.files["audio_file"]
		audio_file_path = "/path/to/save/audio.wav"  # Adjust as needed
		audio_file.save(audio_file_path)
    
    	# Transcribe audio file
		context = transcribe_audio(audio_file_path)
		answer = 'not found, please try again!'
		if flask.request.form.get("btn") == 'ratio':
			result = model(context, ratio=0.3)
			answer = ''.join(result)
		elif flask.request.form.get("btn") == 'sentence':
			result = model(context, num_sentences=5)
			answer = ''.join(result)
		return render_template("index.html", answer=answer, context=context)

	elif flask.request.method == "GET":
		return render_template("index.html")

	else:
		flash("No content to summarize!", "warning")
		return render_template("index.html")
	

# Web Scraping Pkg
from bs4 import BeautifulSoup
# from urllib.request import urlopen
from urllib.request import urlopen

# Sumy Pkg
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

def sumy_summary(docx):
	parser = PlaintextParser.from_string(docx,Tokenizer("english"))
	lex_summarizer = LexRankSummarizer()
	summary = lex_summarizer(parser.document,3)
	summary_list = [str(sentence) for sentence in summary]
	result = ' '.join(summary_list)
	return result


# Reading Time
def readingTime(mytext):
	total_words = len([ token.text for token in nlp(mytext)])
	estimatedTime = total_words/200.0
	return estimatedTime

# Fetch Text From Url
def get_text(url):
	page = urlopen(url)
	soup = BeautifulSoup(page)
	fetched_text = ' '.join(map(lambda p:p.text,soup.find_all('p')))
	return fetched_text

@app.route('/compare_summary')
def compare_summary():
	return render_template('compare_summary.html')

@app.route('/comparer',methods=['GET','POST'])
def comparer():
	start = time.time()
	if request.method == 'POST':
		# Transcribe audio file
		
		audio_file = flask.request.files["audio_file"]
		audio_file_path = "audio.wav"  # Adjust as needed
		audio_file.save(audio_file_path)
		rawtext = transcribe_audio(audio_file_path)
		print(rawtext)
		answer = 'not found, please try again!'
		final_reading_time = readingTime(rawtext)
		final_summary_spacy, charts = text_summarizer(rawtext)
		summary_reading_time = readingTime(final_summary_spacy)
		
		# NLTK
		final_summary_nltk = nltk_summarizer(rawtext)
		summary_reading_time_nltk = readingTime(final_summary_nltk)
		# Sumy
		final_summary_sumy = sumy_summary(rawtext)
		summary_reading_time_sumy = readingTime(final_summary_sumy) 

		end = time.time()
		final_time = end-start

		import matplotlib
		matplotlib.use('agg')
		from matplotlib import pyplot as plt
		import mpld3

		fig, ax = plt.subplots()
		model_names = ("SPACY", "NLTK", "SUMY LEXRANK")
		summary_time_cmp = [summary_reading_time,summary_reading_time_nltk, summary_reading_time_sumy]
		ax.bar([0,1,2], summary_time_cmp, align='center', alpha=0.5)
		plt.xticks([0,1,2], model_names, rotation='vertical')
		plt.ylabel('Time in minute')
		plt.title('Time comparision between different models')
		chart_html1 = mpld3.fig_to_html(fig)

	return render_template('compare_summary.html',ctext=rawtext,final_summary_spacy=final_summary_spacy,
		final_summary_nltk=final_summary_nltk,final_time=final_time,
		final_reading_time=final_reading_time,summary_reading_time=summary_reading_time,
		final_summary_sumy=final_summary_sumy,
		summary_reading_time_sumy=summary_reading_time_sumy,summary_reading_time_nltk=summary_reading_time_nltk,
		chart_html_array=[chart_html1])

if __name__ == '__main__':
	print("Starting web service")
	app.run(host='127.0.0.1', port="5000", debug=True)




