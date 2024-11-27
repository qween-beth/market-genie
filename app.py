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
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///marketgenie2.db'
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
    username = db.Column(db.String(150), nullable=False, unique=True, index=True)
    email = db.Column(db.String(150), nullable=False, unique=True, index=True)
    password = db.Column(db.String(150), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    contents = db.relationship('SegmentContent', backref='author', lazy=True)

    def to_dict(self):
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "created_at": self.created_at
        }

class Segment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False, unique=True, index=True)
    description = db.Column(db.String(255))
    members = db.relationship('SegmentMember', backref='segment', lazy='dynamic', cascade="all, delete-orphan")
    contents = db.relationship('SegmentContent', backref='related_segment', lazy='dynamic', cascade="all, delete-orphan")

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description
        }

class SegmentMember(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), nullable=False)
    email = db.Column(db.String(150), nullable=False, index=True)
    segment_id = db.Column(
        db.Integer,
        db.ForeignKey('segment.id', ondelete='CASCADE'),
        nullable=False,
        index=True
    )
    added_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "email": self.email,
            "segment_id": self.segment_id,
            "added_at": self.added_at
        }

class SegmentContent(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    segment_id = db.Column(
        db.Integer,
        db.ForeignKey('segment.id', ondelete='CASCADE'),
        nullable=False,
        index=True
    )
    user_id = db.Column(
        db.Integer,
        db.ForeignKey('user.id', ondelete='CASCADE'),
        nullable=False,
        index=True
    )
    content_type = db.Column(db.String(50), nullable=False)
    context = db.Column(db.String(500))
    text = db.Column(db.Text)
    model_provider = db.Column(db.String(50), default='openai')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_modified = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "segment_id": self.segment_id,
            "user_id": self.user_id,
            "content_type": self.content_type,
            "context": self.context,
            "text": self.text,
            "model_provider": self.model_provider,
            "created_at": self.created_at,
            "last_modified": self.last_modified
        }


def generate_segment_content(segment_name, content_type, context=None, model_provider='openai'):
    """
    Generate content for an entire segment with a single, segment-specific generation.
    Supports multiple model providers like OpenAI, Groq, and Claude.
    """
    base_context = f"You are an expert marketing copywriter creating content for the {segment_name} segment."
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
        "prompt": f"{base_context} Write engaging marketing content for this segment.",
        "specifics": "Ensure the content is targeted and relevant to the segment's profile."
    })

    final_prompt = f"{content_config['prompt']}\n\nSpecific Requirements:\n{content_config['specifics']}"

    try:
        if model_provider == 'openai':
            response = openai.ChatCompletion.create(
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
                max_tokens=350
            )
            return response.choices[0].message.content.strip()

        elif model_provider == 'claude':
            anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
            if not anthropic_api_key:
                raise ValueError("Anthropic API key not found in environment variables.")
            client = Anthropic(api_key=anthropic_api_key)
            response = client.messages.create(
                model="claude-3-5-sonnet-20240620",
                max_tokens=350,
                messages=[{"role": "user", "content": final_prompt}]
            )
            return response.content[0].text.strip()

        else:
            return "Unsupported model provider."

    except Exception as e:
        return f"Error generating content: {str(e)}"

def process_segment_file(file_path, user_id):
    """
    Process an uploaded CSV file and add segments and members to the database.
    """
    try:
        df = pd.read_csv(file_path)
        required_columns = ['Segment', 'Name', 'Email']
        if not all(col in df.columns for col in required_columns):
            raise ValueError('CSV must contain Segment, Name, and Email columns.')

        segment_groups = df.groupby('Segment')
        for segment_name, segment_df in segment_groups:
            segment = Segment.query.filter_by(name=segment_name).first()
            if not segment:
                segment = Segment(name=segment_name)
                db.session.add(segment)
                db.session.flush()

            for _, row in segment_df.iterrows():
                existing_member = SegmentMember.query.filter_by(
                    name=row['Name'],
                    email=row['Email'],
                    segment_id=segment.id
                ).first()
                if not existing_member:
                    member = SegmentMember(
                        name=row['Name'],
                        email=row['Email'],
                        segment_id=segment.id
                    )
                    db.session.add(member)

        db.session.commit()
        return 'Segments and members successfully imported!'

    except Exception as e:
        db.session.rollback()
        return f'Error processing file: {str(e)}'




@app.route('/')
def home():
    if current_user.is_authenticated:
        recent_contents = SegmentContent.query.filter_by(user_id=current_user.id)\
            .order_by(SegmentContent.created_at.desc())\
            .limit(5).all()
    
    return render_template('home.html')

@app.route('/dashboard')
@login_required  # Ensure the user is logged in
def dashboard():
    # Fetch recent contents authored by the current user
    recent_contents = SegmentContent.query.filter_by(user_id=current_user.id)\
        .order_by(SegmentContent.created_at.desc())\
        .limit(5).all()

    return render_template('dashboard.html', recent_contents=recent_contents)


@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/segments')
@login_required
def list_segments():
    segments = Segment.query.all()
    return render_template('segments.html', segments=segments)

@app.route('/segment/<int:segment_id>')
@login_required
def view_segment(segment_id):
    segment = Segment.query.get_or_404(segment_id)
    users = segment.members.all()
    contents = segment.contents.order_by(SegmentContent.created_at.desc()).all()
    return render_template('segment_detail.html', segment=segment, users=users, contents=contents)

@app.route('/generate_segment_content/<int:segment_id>', methods=['POST'])
@login_required
def generate_segment_content_route(segment_id):
    segment = Segment.query.get_or_404(segment_id)
    content_type = request.form.get('content_type', 'email')
    context = request.form.get('context', '').strip()
    model_provider = request.form.get('model_provider', 'openai')

    content_text = generate_segment_content(
        segment.name,
        content_type,
        context,
        model_provider
    )

    segment_content = SegmentContent(
        segment_id=segment.id,
        user_id=current_user.id,
        content_type=content_type,
        context=context,
        text=content_text,
        model_provider=model_provider
    )
    db.session.add(segment_content)
    db.session.commit()

    flash(f'Content generated for {segment.name} segment.', 'success')
    return redirect(url_for('view_segment', segment_id=segment.id))

@app.route('/content/<int:content_id>/view', methods=['GET'])
@login_required
def view_content(content_id):
    content = SegmentContent.query.get_or_404(content_id)
    
    # Ensure the user can only view their own content
    if content.user_id != current_user.id:
        flash('You are not authorized to view this content.', 'error')
        return redirect(url_for('results'))
    
    return render_template('content_view.html', content=content)

@app.route('/content/<int:content_id>/edit', methods=['GET', 'POST'])
@login_required
def edit_content(content_id):
    content = SegmentContent.query.get_or_404(content_id)
    
    # Ensure the user can only edit their own content
    if content.user_id != current_user.id:
        flash('You are not authorized to edit this content.', 'error')
        return redirect(url_for('results'))
    
    if request.method == 'POST':
        # Update the content
        content.text = request.form.get('content_text')
        content.context = request.form.get('context', '')
        
        try:
            db.session.commit()
            flash('Content updated successfully.', 'success')
            return redirect(url_for('view_segment', segment_id=content.segment_id))
        except Exception as e:
            db.session.rollback()
            flash(f'Error updating content: {str(e)}', 'error')
    
    return render_template('content_edit.html', content=content)

@app.route('/content/<int:content_id>/delete', methods=['POST'])
@login_required
def delete_content(content_id):
    content = SegmentContent.query.get_or_404(content_id)
    
    # Ensure the user can only delete their own content
    if content.user_id != current_user.id:
        flash('You are not authorized to delete this content.', 'error')
        return redirect(url_for('results'))
    
    try:
        db.session.delete(content)
        db.session.commit()
        flash('Content deleted successfully.', 'success')
        return redirect(url_for('view_segment', segment_id=content.segment_id))
    except Exception as e:
        db.session.rollback()
        flash(f'Error deleting content: {str(e)}', 'error')
        return redirect(url_for('view_segment', segment_id=content.segment_id))
    
    
@app.route('/results')
@login_required
def results():
    page = max(1, request.args.get('page', 1, type=int))
    per_page = max(1, min(request.args.get('per_page', 10, type=int), 100))  # Limit max items per page to 100

    contents = SegmentContent.query.filter_by(user_id=current_user.id) \
        .order_by(SegmentContent.created_at.desc()) \
        .paginate(page=page, per_page=per_page)

    return render_template('results.html', contents=contents)

@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            file_path = secure_filename(file.filename)
            file.save(file_path)
            result = process_segment_file(file_path, current_user.id)
            os.remove(file_path)  # Clean up
            flash(result, 'success')
            return redirect(url_for('list_segments'))

    return render_template('upload.html')

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

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

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)