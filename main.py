import os
import time
import pandas as pd
import pathway as pw
from pathway.xpacks.llm.vector_store import VectorStoreServer, VectorStoreClient
from pathway.xpacks.llm import embedders, splitters, parsers
import google.generativeai as genai

# ==========================================
# CONFIGURATION
# ==========================================
# 1. PASTE YOUR GOOGLE KEY HERE
GOOGLE_KEY = "PASTE_YOUR_GOOGLE_KEY_HERE"

# 2. FILE SETUP
DATA_FOLDER = "./data"
CSV_FILE = "test.csv"

# ==========================================
# PART 1: THE PATHWAY SERVER
# ==========================================
def start_server():
    print("--- 1. Initializing Pathway ---")
    
    # Read files
    data_sources = [
        pw.io.fs.read(path=DATA_FOLDER, format="binary", with_metadata=True)
    ]
    
    # Split text
    text_splitter = splitters.TokenCountSplitter(min_tokens=200, max_tokens=400)
    
    # Free local embedder
    print("Loading embedding model...")
    embedder = embedders.SentenceTransformerEmbedder("all-MiniLM-L6-v2")

    # Start Server
    vector_server = VectorStoreServer(
        *data_sources,
        embedder=embedder,
        splitter=text_splitter,
        parser=parsers.UnstructuredParser()
    )
    
    vector_server.run_server(host="127.0.0.1", port=8000, threaded=True)
    print("Server started! Indexing novels... (Waiting 20s)")
    time.sleep(20)
    return vector_server

# ==========================================
# PART 2: THE REASONING LOOP
# ==========================================
def run_analysis():
    print("--- 2. Connecting to Gemini ---")
    
    os.environ["GOOGLE_API_KEY"] = GOOGLE_KEY
    genai.configure(api_key=GOOGLE_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
    client = VectorStoreClient(host="127.0.0.1", port=8000)
    
    df = pd.read_csv(CSV_FILE)
    results = []

    total = len(df)
    for index, row in df.iterrows():
        story_id = row['id']          # CORRECTED COLUMN NAME
        backstory = row['content']    # CORRECTED COLUMN NAME
        
        print(f"[{index+1}/{total}] Checking Story ID {story_id}...")
        
        try:
            relevant_chunks = client.query(backstory, k=3)
            evidence = "\n---\n".join([c['text'] for c in relevant_chunks])
        except:
            evidence = "Search failed."

        prompt = f"""
        You are checking a novel for plot holes.
        EVIDENCE: {evidence}
        BACKSTORY: "{backstory}"
        TASK:
        1. Does the backstory contradict the evidence?
        2. Output strict format: Label | Rationale
        (1 = Consistent, 0 = Contradict)
        """
        
        try:
            response = model.generate_content(prompt)
            output = response.text.strip()
            
            if "|" in output:
                label, rationale = output.split("|", 1)
            else:
                label = "0" if "0" in output else "1"
                rationale = output
                
            results.append([story_id, int(label.strip()), rationale.strip()])
            
        except:
            results.append([story_id, 1, "Error"])

    output_df = pd.DataFrame(results, columns=["id", "prediction", "rationale"])
    output_df.to_csv("results.csv", index=False)
    print("Done!")

if __name__ == "__main__":
    start_server()
    run_analysis()