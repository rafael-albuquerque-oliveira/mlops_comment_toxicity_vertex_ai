import requests
from flask import Flask, request, jsonify
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import google.auth
from google.auth.transport.requests import Request

# Configuração do Flask
app = Flask(__name__)

# Configuração do Google Cloud
PROJECT_ID = "mlops-ufscar"  # Substitua pelo seu ID do projeto
ENDPOINT_ID = "3824252074497933312"  # Substitua pelo seu Endpoint ID
LOCATION = "us-central1"
MODEL_ID = "8469807449107333120"  # Substitua pelo Model ID correto

# Carregar modelo de tradução do Hugging Face
class Translator:
    def __init__(self, model_name="unicamp-dl/translation-pt-en-t5"):
        print("Carregando modelo de tradução...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    def translate(self, text: str) -> str:
        """Traduz um texto do Português para o Inglês."""
        print("Traduzindo texto...")
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        outputs = self.model.generate(**inputs)
        translated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translated_text

translator = Translator()

def get_access_token():
    """Obtém um token de autenticação do Google Cloud para acessar o Vertex AI."""
    credentials, _ = google.auth.default()
    credentials.refresh(Request())
    return credentials.token

def predict_vertex_ai(text):
    """Envia um texto para o modelo no Vertex AI e retorna a previsão."""
    url = f"https://{LOCATION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{LOCATION}/endpoints/{ENDPOINT_ID}:predict"
    
    headers = {
        "Authorization": f"Bearer {get_access_token()}",
        "Content-Type": "application/json"
    }

    data = {"instances": [text]}

    response = requests.post(url, headers=headers, json=data)
    return response.json()

@app.route("/predict", methods=["POST"])
def predict():
    """Recebe um comentário, traduz para inglês e retorna a previsão de toxicidade."""
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    text = data["text"]
    
    # Traduzir texto do Português para Inglês
    translated_text = translator.translate(text)

    # Enviar para o modelo no Vertex AI
    prediction = predict_vertex_ai(translated_text)

    return jsonify({
        "original_text": text,
        "translated_text": translated_text,
        "prediction": prediction
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
