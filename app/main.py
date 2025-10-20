from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
from dotenv import load_dotenv
from .document_processor import DocumentProcessor
from .ai_analyzer import AIAnalyzer

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key')
app.config['UPLOAD_FOLDER'] = os.getenv('UPLOAD_FOLDER', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = int(os.getenv('MAX_FILE_SIZE', 16777216))

ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'docx', 'txt'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

doc_processor = DocumentProcessor()
ai_analyzer = AIAnalyzer()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            text = doc_processor.extract_text(filepath)
            
            if not text or len(text.strip()) < 10:
                return jsonify({'error': 'Could not extract text from document'}), 400
            
            summary = ai_analyzer.summarize_text(text)
            key_points = ai_analyzer.extract_key_points(text)
            suggested_questions = ai_analyzer.get_suggested_questions(text)
            
            return jsonify({
                'success': True,
                'filename': filename,
                'text': text[:1000] + '...' if len(text) > 1000 else text,
                'full_text': text,
                'summary': summary,
                'key_points': key_points,
                'suggested_questions': suggested_questions,
                'word_count': len(text.split())
            })
        
        except Exception as e:
            return jsonify({'error': f'Error processing document: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    question = data.get('question', '')
    context = data.get('context', '')
    
    if not question or not context:
        return jsonify({'error': 'Question and context required'}), 400
    
    try:
        result = ai_analyzer.answer_question(question, context)
        return jsonify({
            'success': True,
            'answer': result['answer'],
            'confidence': result['confidence'],
            'explanation': result['explanation'],
            'context_used': result.get('context_used', '')
        })
    except Exception as e:
        return jsonify({'error': f'Error answering question: {str(e)}'}), 500

@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    data = request.get_json()
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'Text required'}), 400
    
    try:
        sentiment = ai_analyzer.analyze_sentiment(text)
        return jsonify({
            'success': True,
            'sentiment': sentiment
        })
    except Exception as e:
        return jsonify({'error': f'Error analyzing sentiment: {str(e)}'}), 500

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 10000))
    debug_mode = os.getenv('FLASK_ENV') == 'development'
    app.run(host='0.0.0.0', port=port, debug=debug_mode)