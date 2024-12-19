UPLOAD_FOLDER = 'cv-library'  # Folder to store CVs
ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx', 'jpeg', 'jpg', 'png'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS