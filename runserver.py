import sys

from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# sys.path.append("relation-extraction/")

from relation_extraction.relextserver.server import relext

app.register_blueprint(relext, url_prefix="/relation-extraction")
