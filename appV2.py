import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from flask import Flask, request, Response, render_template_string
import requests

OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama2:latest"

app = Flask(__name__)

# Inicializa ChromaDB y modelo de embeddings
chroma_client = chromadb.Client(Settings(persist_directory="./chroma_db"))
collection = chroma_client.get_or_create_collection("itil_docs")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

@app.route('/')
def home():
    return render_template_string(r'''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Chat Stream</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                #chat { max-width: 800px; margin: auto; }
                #messages { height: 400px; border: 1px solid #ccc; padding: 10px; overflow-y: auto; margin-bottom: 10px; }
                .message { margin: 5px 0; padding: 5px; border-radius: 5px; }
                .user { background-color: #e3f2fd; }
                .assistant-thinking { background-color: #fff9c4; font-style: italic; }
                .assistant { background-color: #f5f5f5; }
                #input { width: calc(100% - 80px); padding: 8px; }
                button { width: 70px; padding: 8px; }
                h1 { text-align: center; color: #333; }
            </style>
        </head>
        <body>
            <h1>© Copyright Notice 2025, Khipus.ai - All Rights Reserved</h1>
            <h1>Practico Final : ChatGPT - ITIL </h1>
            <h1>Team Itil           : Andres Moron - Javier Castellon Galvis - Huascar Rivero Lara</h1>
            <div id="chat">
                <div id="messages"></div>
                <form onsubmit="sendMessage(event)">
                    <input type="text" id="input" placeholder="Type your message...">
                    <button type="submit">Send</button>
                </form>
            </div>
          <script>
    async function sendMessage(event) {
        event.preventDefault();
        const input = document.getElementById('input');
        const userMsg = input.value;
        input.value = '';
        appendMessage("You: " + userMsg, "user");

        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: userMsg })
            });

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let responseContent = "";
            let responseMessageElement = null;

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                const chunk = decoder.decode(value, { stream: true });
                const lines = chunk.split('\n').filter(Boolean);
                for (const line of lines) {
                    try {
                        const data = JSON.parse(line);
                        if (data.response) {
                            responseContent += data.response;
                            if (!responseMessageElement) {
                                responseMessageElement = document.createElement('div');
                                responseMessageElement.className = 'message assistant';
                                document.getElementById('messages').appendChild(responseMessageElement);
                            }
                            responseMessageElement.textContent = "Assistant: " + responseContent;
                        }
                    } catch (e) {
                        // Si no es JSON, ignora
                    }
                }
            }
        } catch (error) {
            console.error('Error:', error);
            appendMessage("Assistant: Failed to get response", "assistant");
        }
    }

    function appendMessage(content, className) {
        const messagesDiv = document.getElementById('messages');
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message ' + className;
        messageDiv.textContent = content;
        messagesDiv.appendChild(messageDiv);
        messagesDiv.scrollTop = messagesDiv.scrollHeight;
    }
</script>
        </body>
        </html>
    ''')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data['message']

    # Recupera contexto relevante de la DB local
    query_emb = embedder.encode([user_message])[0].tolist()
    results = collection.query(query_embeddings=[query_emb], n_results=1)
    # Verifica si hay resultados
    if not results["documents"] or not results["documents"][0]:
        contexto = "No se encontró información relevante en la base de datos."
    else:
        contexto = results["documents"][0][0][:1000]  # Limita el contexto

    system_message = (
        "You are a helpful assistant. Responde SIEMPRE en español. Incluye razonamiento interno envuelto en <think>...</think> antes de la respuesta final."
    )
    prompt = f"{system_message}\n\nContexto:\n{contexto}\n\nUser: {user_message}\nAssistant:"

    def generate():
        try:
            response = requests.post(
                OLLAMA_API_URL,
                json={
                    "model": MODEL_NAME,
                    "prompt": prompt,
                    "stream": True
                },
                stream=True,
                timeout=60
            )
            for line in response.iter_lines(decode_unicode=True):
                if line:
                    yield line
        except Exception as e:
            yield f"Assistant: Error: {str(e)}"

    return Response(generate(), mimetype='text/plain')

if __name__ == '__main__':
    app.run(debug=True)