import argparse
from flask import Flask, render_template, request
from logging.config import dictConfig

dictConfig(
    {
        "version": 1,
        "formatters": {
            "default": {
                "format": "[%(asctime)s] %(levelname)s in %(module)s: %(message)s",
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
                "formatter": "default",
            },
            "file": {
                "class": "logging.FileHandler",
                "filename": "service/flask.log",
                "formatter": "default",
            },
        },
        "root": {"level": "DEBUG", "handlers": ["console", "file"]},
    }
)

app = Flask(__name__)

from flask_cors import CORS
cors = CORS(app, resources={r"/api/*": {"origins": "http://localhost:5173"}})

import dotenv
config = dotenv.dotenv_values('.env')

from flask_httpauth import HTTPTokenAuth
auth = HTTPTokenAuth(scheme='Bearer')
tokens = { config['APP_TOKEN']: "user1", }

@auth.verify_token
def verify_token(token):
    if token in tokens:
        return tokens[token]

import joblib

# Сохранение модели
MODEL_NAME = "models/gradient_boost_v1.pkl"


# Маршрут для отображения формы
@app.route("/")
def index():
    return render_template("index.html")


# Маршрут для обработки данных формы
@app.route("/api/numbers", methods=["POST"])
def process_numbers():

    data = request.get_json()

    app.logger.info(f"Requst data: {data}")
    
    
    try:
        area = float(data["area"])
        rooms = int(data['rooms'])
        total_floors = int(data['total_floors'])
        floor = int(data['floor'])
        
        print(area, rooms, total_floors, floor)
    
                
        price = app.config["model"].predict([[area, rooms, total_floors, floor]])[0]
        price = int(price)
    except ValueError:
        return {"status": "error", "data": "Ошибка парсинга данных"}
    return {"status": "success", "data": price}


if __name__ == "__main__":
    """Parse arguments and run lifecycle steps"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help="Model name", default=MODEL_NAME)
    args = parser.parse_args()

    app.config["model"] = joblib.load(args.model)
    app.logger.info(f"Use model: {args.model}")
    app.run(debug=True)
