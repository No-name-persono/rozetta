
import os, sys
# Ensure the project root (this file's directory) is on sys.path
_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

from flask import Flask
from config import DEFAULTS
from blueprints.qa_calls import qa_calls_bp
from blueprints.admin import admin_bp
from blueprints.speaker_lab import speaker_lab_bp
from blueprints.api import api_bp


def create_app():
    app = Flask(__name__)
    app.secret_key = DEFAULTS.get("SECRET_KEY", "dev")
    app.config['MAX_CONTENT_LENGTH'] = DEFAULTS["MAX_UPLOAD_MB"] * 1024 * 1024

    app.register_blueprint(qa_calls_bp)
    app.register_blueprint(admin_bp)
    app.register_blueprint(speaker_lab_bp)
    app.register_blueprint(api_bp)

    return app


if __name__ == "__main__":
    app = create_app()
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "5010"))
    debug = os.getenv("FLASK_DEBUG", "0") in {"1", "true", "True"}
    app.run(host=host, port=port, debug=debug)
