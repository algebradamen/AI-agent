import os
import json
from flask import Flask, render_template, request, jsonify
import geopandas as gpd
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent

# --- KONFIGURASJON ---
app = Flask(__name__)

# Last inn API-nøkkelen
with open("tangen-key.txt", "r", encoding="utf-8") as f:
    os.environ["OPENAI_API_KEY"] = f.read().strip()

# Last inn data (GEOJSON)
gdf = gpd.read_file("test2.geojson")

# Opprett agenten
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
agent = create_pandas_dataframe_agent(
    llm,
    gdf,
    verbose=False,  # Setter til False for web-bruk
    allow_dangerous_code=True,
    agent_type="openai-tools",
)

# --- RUTER ---

@app.route("/")
def index():
    """Serverer hovedsiden"""
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    """Mottar spørsmål fra bruker og returnerer svar fra agenten"""
    try:
        data = request.json
        user_input = data.get("question", "").strip()
        
        if not user_input:
            return jsonify({"error": "Spørsmål kan ikke være tomt"}), 400
        
        # Kjør agenten
        response = agent.invoke(user_input)
        answer = response.get("output", "Ingen svar fra agenten")
        
        return jsonify({"answer": answer}), 200
    
    except Exception as e:
        return jsonify({"error": f"En feil oppstod: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
