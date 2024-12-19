from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename
from get_cv_upload_response import query_ragcv, chat,gen_feedback
from extract_details import populate_dbcv, clear_vector_db
from flask_cors import CORS
import os
import uuid
import logging
from PIL import Image
import pytesseract
from pdf2image import convert_from_path
from docx import Document as DocxDocument
from sql_chat import return_low_matched_jobs,return_intermediate_matched_jobs
from pdf_upload_configs import ALLOWED_EXTENSIONS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://localhost:3000", "http://localhost:3001", "http://localhost:3002"]}})
UPLOAD_FOLDER = 'cv-library'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx', 'jpeg', 'jpg', 'png'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_text_from_file(file_path, file_type):
    text = ""
    try:
        if file_type == 'application/pdf':
            images = convert_from_path(file_path)
            for image in images:
                text += pytesseract.image_to_string(image)
        elif file_type in ['application/msword', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document']:
            doc = DocxDocument(file_path)
            for para in doc.paragraphs:
                text += para.text
        elif file_type.startswith('image/'):
            image = Image.open(file_path)
            text = pytesseract.image_to_string(image)
    except Exception as e:
        logging.error(f"Error extracting text from file {file_path}: {e}")
        raise
    return text

@app.route('/upload-cv', methods=['POST'])
def upload_cv():
    try:
        if not os.path.exists(UPLOAD_FOLDER):
            os.makedirs(UPLOAD_FOLDER)
        file = request.files['file']
        user_id = request.form['userID']
        if not file:
            app.logger.error("No file provided")
            return jsonify({'error': 'No file provided'}), 400
        if not user_id:
            app.logger.error("No user ID provided")
            return jsonify({'error': 'No user ID provided'}), 400

        app.logger.info(f"Received file: {file.filename} from user: {user_id}")

        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}_{user_id}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)

        file_type = file.content_type
        app.logger.info(f"File type: {file_type}")
        if not allowed_file(file.filename):
            app.logger.error("File type not allowed")
            os.remove(file_path)
            return jsonify({'error': 'File type not allowed'}), 400

        text = extract_text_from_file(file_path, file_type)

        clear_vector_db()
        documents = populate_dbcv([text])
        add_db_message = documents[1]

        os.remove(file_path)

        return jsonify({'success': 'File successfully uploaded', 'db_message': add_db_message}), 200
    except Exception as e:
        app.logger.error(f"Error uploading file: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/query-cv', methods=['POST'])
def query_cv():
    try:
        response = query_ragcv()
        return jsonify({'response': response}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/clear-db', methods=['GET'])
def clear_db():
    try:
        clear_vector_db()
        return jsonify({'success': 'Vector DB cleared'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/chat', methods=['POST'])
def query():
    data = request.json
    question = data.get('question')
    print(question)

    if not question:
        return jsonify({"error": "Question parameter is required"}), 400

    try:
        result = return_low_matched_jobs(question)
        return jsonify({"result": result}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/chat_high', methods=['POST'])
def query_high():
    data = request.json
    question = data.get('question')
    experience = data.get('experience')
    print(question)
    print(experience)

    if not question:
        return jsonify({"error": "Question parameter is required"}), 400

    try:
        result = return_intermediate_matched_jobs(question,experience)
        return jsonify({"result": result}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/chat_ui', methods=['POST'])
def query_chat():
    data = request.json
    question = data.get('question')
    details = data.get('details')
    print(question)
    print(details)

    if not question:
        return jsonify({"error": "Question parameter is required"}), 400

    try:
        result = chat(question, details)
        return jsonify({"result": result}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route('/feedback',methods=['POST'])
def query_feedback():
    data = request.json
    skills = data.get('skills')
    print(skills)
    try:
        feedback= gen_feedback(skills)
        return jsonify({'response': feedback}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)