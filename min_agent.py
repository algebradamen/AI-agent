import os
import geopandas as gpd
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
import pandas as pd
from io import StringIO
from io import FileIO

# --- 1. KONFIGURASJON ---
# Du må bytte ut denne med din egen nøkkel fra platform.openai.com
# os.environ["OPENAI_API_KEY"] = FileIO.readall("secret-key.txt").strip()

with open("secret-key.txt", "r", encoding="utf-8") as f:
    os.environ["OPENAI_API_KEY"] = f.read().strip()


# --- 2. LAST INN DATA (GEOJSON) ---

# Siden jeg ikke har filen din, lager jeg en liten "jukse-fil" her.
# Når du skal bruke din egen fil, slett linjene under og bruk:
gdf = gpd.read_file("test2.geojson")

# csv_data = """
# navn,type,befolkning,geometry
# Sentrum,Bydel,1500,POINT(10.75 59.91)
# Nordmarka,Skog,50,POINT(10.70 60.00)
# Gamlebyen,Bydel,4000,POINT(10.77 59.90)
# """
# Vi later som dette er en GeoDataFrame for eksempelet
# df = pd.read_csv(StringIO(csv_data))
# gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.navn, df.type)) # Forenklet geometri for demo

# endre_df = gdf[gdf['kriterium_truetArt'].astype(str).str.lower() == 'ja']
# print(endre_df[['vitenskapeligNavn','norskNavn','listeStatus_beskrivelse']])


print("✅ Data lastet inn! Agenten er klar.")

# --- 3. LAG AGENTEN ---

# Vi velger modellen. gpt-4 er best på koding, gpt-3.5 er billigere.
# llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)
llm = ChatOpenAI(model="gpt-5-nano", temperature=0)


# Dette er magien: LangChain har en ferdig agent for tabeller og geodata.
# Den lar LLM-en skrive sin egen Pandas/Python-kode for å finne svar.
agent = create_pandas_dataframe_agent(
    llm,
    gdf,
    verbose=True, # Setter denne til True så du ser hva agenten "tenker"
    allow_dangerous_code=True, # Tillater at agenten kjører python-kode for å analysere data
    agent_type="openai-tools",
)

# --- 4. SNAKK MED AGENTEN ---

print("\n--- Start samtale (Skriv 'slutt' for å avslutte) ---")
print("Eksempel: 'Vis meg oversikten over truede arter?' eller 'Vis meg data for truede arter'")

while True:
    user_input = input("\nDu: ")
    if user_input.lower() in ["slutt", "exit", "quit"]:
        break
    
    try:
        # Agenten tenker og svarer
        response = agent.invoke(user_input)
        print(f"Agent: {response['output']}")
    except Exception as e:
        print(f"En feil oppstod: {e}")
        