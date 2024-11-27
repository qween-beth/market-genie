from flask import Flask, render_template, request, redirect, url_for, send_file, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import openai
from anthropic import Anthropic
from groq import Groq
from datetime import datetime
import os
from werkzeug.utils import secure_filename
import tempfile


app = Flask(__name__)


app.secret_key = os.environ.get('SECRET_KEY', 'dev_key')  # Change in production
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///marketgenie.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Login Manager Setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'


# OpenAI API Setup
openai.api_key = os.environ.get('OPENAI_API_KEY')

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), nullable=False, unique=True)
    email = db.Column(db.String(150), nullable=False, unique=True)
    password = db.Column(db.String(150), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    contents = db.relationship('GeneratedContent', backref='author', lazy=True)

class GeneratedContent(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    customer_name = db.Column(db.String(150), nullable=False)
    segment = db.Column(db.String(150))
    email = db.Column(db.String(150))
    content_type = db.Column(db.String(50), nullable=False)
    context = db.Column(db.String(500))  # New field for storing context
    text = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_modified = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


def generate_marketing_content(segment, first_name, content_type, context=None, model_provider='openai'):
    """
    Generate personalized content using the specified AI provider (OpenAI, Groq, or Claude).
    Includes detailed prompts tailored for various content types.
    """
    # Base context combining customer info and optional context
    base_context = f"You are an expert marketing copywriter creating content for {first_name} who belongs to the {segment} segment."
    if context:
        base_context += f" The specific context/campaign is: {context}."

    # Enhanced prompts with specific instructions per content type
    prompts = {
        "email": {
            "prompt": f"{base_context} Write a persuasive marketing email that drives engagement and conversion.",
            "specifics": "Include: Subject line (marked with 'Subject:'), greeting, body, and call-to-action. Keep it concise and personal."
        },
        "social_post": {
            "prompt": f"{base_context} Create an engaging social media post that generates high engagement.",
            "specifics": "Include: Main message, relevant hashtags, and a clear call-to-action. Optimize for sharing and engagement."
        },
        "blog_post": {
            "prompt": f"{base_context} Write an informative blog post introduction that hooks readers.",
            "specifics": "Include: Attention-grabbing headline, compelling introduction paragraph, and clear value proposition."
        },
        "video_script": {
            "prompt": f"{base_context} Create a compelling video script that captures attention in the first 10 seconds.",
            "specifics": "Include: Opening hook, key messages, and clear instructions for visuals and audio."
        },
        "ad_copy": {
            "prompt": f"{base_context} Write compelling ad copy that drives conversions.",
            "specifics": "Include: Headline, main copy, and strong call-to-action. Focus on benefits and urgency."
        }
    }

    content_config = prompts.get(content_type, {
        "prompt": f"{base_context} Write engaging marketing content.",
        "specifics": "Ensure the content is engaging and relevant to the audience."
    })

    final_prompt = f"{content_config['prompt']}\n\nSpecific Requirements:\n{content_config['specifics']}"

    try:
        if model_provider == 'openai':
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": final_prompt}],
                max_tokens=350,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()

        elif model_provider == 'groq':
            client = Groq(api_key=os.getenv('GROQ_API_KEY'))
            response = client.chat.completions.create(
                model="mixtral-8x7b-32768",
                messages=[{"role": "system", "content": final_prompt}],
                max_tokens=350,
                temperature=0.7,
                top_p=1.0,
                frequency_penalty=0.2,
                presence_penalty=0.0
            )
            return response.choices[0].message.content.strip()

        elif model_provider == 'claude':
            anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
            if not anthropic_api_key:
                raise ValueError("Anthropic API key not found in environment variables.")
            
            # Initialize the Anthropic client
            client = Anthropic(api_key=anthropic_api_key)
            
            # Create a message using the Messages API
            response = client.messages.create(
                model="claude-3-5-sonnet-20240620",
                max_tokens=350,
                temperature=0.7,
                messages=[
                    {
                        "role": "user",
                        "content": final_prompt
                    }
                ]
            )
            return response.content[0].text.strip()

        else:
            return "Unsupported model provider."

    except Exception as e:
        return f"Error generating content: {str(e)}"


def process_file(file_path, user_id, content_type, context, model_provider):
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        required_columns = ['Customer Name', 'Segment', 'Email']

        if not all(col in df.columns for col in required_columns):
            raise ValueError('CSV must contain Customer Name, Segment, and Email columns')

        # Group users by segment
        segment_groups = df.groupby('Segment')
        contents_created = 0

        # Generate content for each segment
        for segment, segment_df in segment_groups:
            # Take the first name from the first user in the segment for prompt personalization
            first_name = segment_df['Customer Name'].iloc[0].split()[0]

            # Generate segment-specific content
            segment_content_text = generate_marketing_content(
                segment,
                first_name,
                content_type,
                context,
                model_provider=model_provider
            )

            # Create content entries for each user in the segment
            for _, row in segment_df.iterrows():
                # Personalize the content with individual user's name
                personalized_content = segment_content_text.replace(
                    first_name, 
                    row['Customer Name'].split()[0]
                )

                content = GeneratedContent(
                    user_id=user_id,
                    customer_name=row['Customer Name'],
                    segment=segment,
                    email=row['Email'],
                    content_type=content_type,
                    context=context,
                    text=personalized_content
                )
                db.session.add(content)
                contents_created += 1

        db.session.commit()
        return f'Successfully generated {contents_created} pieces of content across {len(segment_groups)} segments!'
    except Exception as e:
        return f'Error processing file: {str(e)}'

    

@app.route('/')
def home():
    if current_user.is_authenticated:
        recent_contents = GeneratedContent.query.filter_by(user_id=current_user.id)\
            .order_by(GeneratedContent.created_at.desc())\
            .limit(5).all()
        return render_template('home.html', recent_contents=recent_contents)
    return render_template('home.html')

        

@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file uploaded', 'error')
            return redirect(request.url)

        file = request.files['file']
        content_type = request.form.get('content_type', 'email')
        context = request.form.get('context', '').strip()  # Get context from form
        model_provider = request.form.get('model_provider', 'openai')  # Default to OpenAI

        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)

        # Validate file extension
        if not file.filename.lower().endswith('.csv'):
            flash('Only CSV files are allowed', 'error')
            return redirect(request.url)

        try:
            # Save the file temporarily for processing
            file_path = os.path.join(tempfile.gettempdir(), secure_filename(file.filename))
            file.save(file_path)

            # Process file asynchronously
            result = process_file(file_path, current_user.id, content_type, context, model_provider)
            flash(result, 'success')
            return redirect(url_for('results'))


        except Exception as e:
            flash(f'Error processing file: {str(e)}', 'error')
            return redirect(request.url)

    return render_template('upload.html')


@app.route('/results')
@login_required
def results():
    page = max(1, request.args.get('page', 1, type=int))
    per_page = max(1, min(request.args.get('per_page', 10, type=int), 100))  # Limit max items per page to 100

    contents = GeneratedContent.query.filter_by(user_id=current_user.id) \
        .order_by(GeneratedContent.created_at.desc()) \
        .paginate(page=page, per_page=per_page)

    return render_template('results.html', contents=contents)


@app.route('/edit_content/<int:content_id>', methods=['GET', 'POST'])
@login_required
def edit_content(content_id):
    content = GeneratedContent.query.get_or_404(content_id)

    if content.user_id != current_user.id:
        flash('Unauthorized access', 'error')
        return redirect(url_for('results'))

    if request.method == 'POST':
        content.text = request.form['content_text']
        content.last_modified = datetime.utcnow()
        db.session.commit()
        flash('Content updated successfully!', 'success')
        return redirect(url_for('results'))

    return render_template('edit_content.html', content=content)

@app.route('/export_csv')
@login_required
def export_csv():
    try:
        data = GeneratedContent.query.filter_by(user_id=current_user.id).all()
        df = pd.DataFrame([{
            'Customer Name': item.customer_name,
            'Segment': item.segment,
            'Email': item.email,
            'Content Type': item.content_type,
            'Context': item.context,  # Include context in export
            'Content': item.text,
            'Created': item.created_at,
            'Last Modified': item.last_modified
        } for item in data])

        export_dir = 'exports'
        if not os.path.exists(export_dir):
            os.makedirs(export_dir)

        filename = f'content_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        filepath = os.path.join(export_dir, filename)
        df.to_csv(filepath, index=False)
        return send_file(filepath, as_attachment=True, download_name=filename)
    except Exception as e:
        flash(f'Error exporting CSV: {str(e)}', 'error')
        return redirect(url_for('results'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')

        if User.query.filter_by(username=username).first():
            flash('Username already exists', 'error')
            return redirect(url_for('register'))

        if User.query.filter_by(email=email).first():
            flash('Email already registered', 'error')
            return redirect(url_for('register'))

        user = User(
            username=username,
            email=email,
            password=generate_password_hash(password, method='scrypt')
        )
        db.session.add(user)
        db.session.commit()

        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('home'))

        flash('Invalid email or password', 'error')
        return redirect(url_for('login'))

    return render_template('login.html')

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)