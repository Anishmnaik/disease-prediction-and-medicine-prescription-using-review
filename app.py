from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, Email, Length, EqualTo, Regexp
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user

import os
import joblib
import pandas as pd
import re
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from sklearn import metrics
import nltk
# import speech_recognition as sr

HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem">{}</div>"""
# import nltk
# nltk.download('wordnet')
app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# -------------------- Models --------------------
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# -------------------- Forms --------------------
class RegisterForm(FlaskForm):
    name = StringField('Username', validators=[DataRequired(), Length(min=3, max=25)])
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[
        DataRequired(),
        Length(min=8),
        Regexp(r'^(?=.*[A-Z])(?=.*\d)(?=.*[\W_]).+$', message='Must include 1 uppercase letter, 1 digit, and 1 special character.')
    ])
    confirm_password = PasswordField('Confirm Password', validators=[DataRequired(), EqualTo('password', message='Passwords must match.')])
    submit = SubmitField('Register')

class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Login')


# Model saved with Keras model.save()
MODEL_PATH = 'passmodel.pkl'


TOKENIZER_PATH = 'tfidfvectorizer.pkl'

DATA_PATH = 'drugsComTrain.csv'

# loading vectorizer
vectorizer = joblib.load(TOKENIZER_PATH)
# loading model
model = joblib.load(MODEL_PATH)

# getting stopwords
stop = stopwords.words('english')
lemmatizer = WordNetLemmatizer()


@app.route('/', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user and check_password_hash(user.password, form.password.data):
            login_user(user)
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid username or password.', 'danger')
    return render_template('login.html', form=form)

@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()
    if form.validate_on_submit():
        existing_user = User.query.filter(
            (User.username == form.name.data) | (User.email == form.email.data)
        ).first()
        if existing_user:
            flash('Username or email already exists.', 'danger')
            return redirect(url_for('register'))

        hashed_password = generate_password_hash(form.password.data)
        new_user = User(username=form.name.data, email=form.email.data, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html', form=form)

@app.route('/logout')
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))


@app.route('/home')
def home():
	return render_template('home.html')

@app.route('/about')
def about():
	return render_template('about.html')

@app.route('/contact')
def contact():
	return render_template('contact.html')

@app.route('/service')
def service():
	return render_template('service.html')

@app.route('/BD')
def about1():
	return render_template('BD.html')

@app.route('/acne')
def about2():
	return render_template('acne.html')

@app.route('/BP')
def about3():
	return render_template('bloodp.html')

@app.route('/depression')
def about4():
	return render_template('depression.html')

@app.route('/diabetes')
def about5():
	return render_template('diabetes.html')

@app.route('/BC')
def about6():
	return render_template('Birth Controll.html')

# @app.route("/logout")
# def logout():
# 	session.clear()
# 	return redirect('/')


# @app.route('/index')
# def index():
# 	if 'user_id' in session:

# 		return render_template('home.html')
# 	else:
# 		return redirect('/')


# @app.route('/login_validation', methods=['POST'])
# def login_validation():
# 	username = request.form.get('username')
# 	password = request.form.get('password')

# 	session['user_id'] = username
# 	session['domain'] = password

# 	if username == "admin@gmail.com" and password == "admin":

# 		return render_template('home.html')
# 		# return render_template('login.html', data=payload)

# 	else:

# 		err = "Kindly Enter valid User ID/ Password"
# 		return render_template('login.html', lbl=err)

# 	return ""


@app.route('/predict', methods=["GET", "POST"])
def predict():
	if request.method == 'POST':
		raw_text = request.form['rawtext']

		if raw_text != "":
			clean_text = cleanText(raw_text)
			clean_lst = [clean_text]

			tfidf_vect = vectorizer.transform(clean_lst)
			prediction = model.predict(tfidf_vect)
			predicted_cond = prediction[0]
			#confidence_pred = metrics.accuracy_score(clean_text, predicted_cond)
			df = pd.read_csv(DATA_PATH)
			top_drugs = top_drugs_extractor(predicted_cond, df)
			nltk.download('vader_lexicon')
			from nltk.sentiment.vader import SentimentIntensityAnalyzer
			sid = SentimentIntensityAnalyzer()
			score = ((sid.polarity_scores(str(clean_text))))['compound']
			if(score > 0):
				labeli = 'This sentence is positive'
			elif(score == 0):
				labeli = 'This sentence is neutral'
			else:
				labeli = 'This sentence is negative'
			return render_template('predict.html', rawtext=raw_text, result=predicted_cond, top_drugs=top_drugs, variable=labeli)
		else:
			 raw_text = "There is no text to select"


def cleanText(raw_review):
    # 1. Delete HTML
    review_text = BeautifulSoup(raw_review, 'html.parser').get_text()
    # 2. Make a space
    letters_only = re.sub('[^a-zA-Z]', ' ', review_text)
    # 3. lower letters
    words = letters_only.lower().split()
    # 5. Stopwords 
    meaningful_words = [w for w in words if not w in stop]
    # 6. lemmitization
    lemmitize_words = [lemmatizer.lemmatize(w) for w in meaningful_words]
    # 7. space join words
    return( ' '.join(lemmitize_words))


def top_drugs_extractor(condition,df):
    df_top = df[(df['rating']>=9)&(df['usefulCount']>=100)].sort_values(by = ['rating', 'usefulCount'], ascending = [False, False])
    drug_lst = df_top[df_top['condition']==condition]['drugName'].head(3).tolist()
    return drug_lst





####SPEECH TO TEXT

# @app.route('/audio_to_text/')
# def audio_to_text():
#     flash(" Press Start to start recording audio and press Stop to end recording audio")
#     return render_template('audio_to_text.html')

# @app.route('/audio', methods=['POST'])
# def audio():
#     r = sr.Recognizer()
    
#     with open('upload/audio.wav', 'wb') as f:
#         f.write(request.data)
  
#     with sr.AudioFile('upload/audio.wav') as source:
#         audio_data = r.record(source)
#         text = r.recognize_google(audio_data, language='en-IN', show_all=True)
#         print(text)
#         return_text = " Did you say : <br> "
#         try:
#             for num, texts in enumerate(text['alternative']):
#                 return_text += str(num+1) +") " + texts['transcript']  + " <br> "
#         except:
#             return_text = " Sorry!!!! Voice not Detected "
        
#         return_text_return = str(return_text)
#         if return_text_return != "":
#             clean_text_return = cleanText(return_text_return)
#             clean_lst_return = [clean_text_return]
#             tfidf_vect_return = vectorizer.transform(clean_lst_return)
#             prediction_return = model.predict(tfidf_vect_return)
#             predicted_cond_return = prediction_return[0]
#             df = pd.read_csv(DATA_PATH)
#             top_drugs = top_drugs_extractor(predicted_cond_return, df)
#             nltk.download('vader_lexicon')
#             from nltk.sentiment.vader import SentimentIntensityAnalyzer
#             sid = SentimentIntensityAnalyzer()
#             score = ((sid.polarity_scores(str(clean_text_return))))['compound']
#             if(score > 0):
#                 labelireturn = 'This sentence is positive'
#             elif(score == 0):
#                 labelireturn = 'This sentence is neutral'
#             else:
#                 labelireturn = 'This sentence is negative'
            
#             return render_template('audio_to_text.html', rawtextreturn=return_text_return, resultreturn=predicted_cond_return, top_drugsreturn=top_drugs, variablereturn=labelireturn)
#         else:
#             return_text_return = "There is no text to select"


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)

