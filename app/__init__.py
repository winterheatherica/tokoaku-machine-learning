from flask import Flask
from dotenv import load_dotenv
import os

def create_app():
    load_dotenv()

    app = Flask(__name__)

    from .routes.sentiment import sentiment_bp
    from .routes.summarize import summarize_bp
    from .routes.sales import sales_bp

    app.register_blueprint(sentiment_bp)
    app.register_blueprint(summarize_bp)
    app.register_blueprint(sales_bp)

    return app
