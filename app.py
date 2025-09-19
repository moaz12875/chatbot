from flask import Flask, render_template, request, jsonify,Blueprint
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import torch
from .model import chatbot_pipeline
from .routes import chatbot_bp
from chatbot import create_app


app = Flask(__name__)

# حمل الموديل من Hugging Face Hub أو من Drive
MODEL_NAME = "https://drive.google.com/file/d/1Noj5rcFzaXtCKv-2JCmU2BuyWXUlfCzn/view?usp=sharing"  # غيّره باللي عندك
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
chatbot = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

chatbot_bp = Blueprint("chatbot", __name__)

@chatbot_bp.route("/")
def index():
    return render_template("index.html")

@chatbot_bp.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    response = chatbot_pipline(user_input, max_length=200, do_sample=True, top_p=0.95, temperature=0.7)
    bot_reply = response[0]["generated_text"]
    return jsonify({"reply": bot_reply})
    
def create_app():
    app = Flask(__name__)
    app.register_blueprint(chatbot_bp)
    return app    
app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
