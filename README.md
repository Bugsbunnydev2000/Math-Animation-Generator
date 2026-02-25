#  Math to Manim – Professional Edition

**Transform natural language math & science descriptions into cinematic Manim animations — 100% local, offline, private.**


## ✨ Highlights

- Completely **local & private** — no API keys, no cloud, no data leaves your machine
- Uses code-specialized LLM (`qwen2.5-coder:7b` recommended)
- Very aggressive **code sanitization & repair** (~40 common Manim pitfalls automatically fixed)
- Clean, modern **Streamlit** interface with quality presets (480p – 4K)
- Generation history, natural-language code refinement, custom resolution/FPS
- Automatic cleanup of old temporary files
- Fallback animation when generation fails


## 🎯 Typical Use Cases

- University lecturers creating step-by-step proof animations
- Math & physics YouTube / TikTok content creators
- Students preparing beautiful presentation videos
- Visual learners who want to *see* derivations, graphs, transformations
- Rapid prototyping of mathematical visualizations without writing Manim code manually

## How It Works : 

User enters natural language description

LLM creates structured, concise animation + math plan

Code-specialized LLM generates strict Manim CE v0.18.1 code

Heavy post-processing & error-fixing of the generated code

Manim renders MP4 video (selected quality preset)

Video preview + download appears in the browser



## 🚀 Quick Start (Windows – 2026)

### Prerequisites (~25–45 min first time)

| Software       | Recommended version / Download                 | Purpose                              | Approx. size |
|----------------|------------------------------------------------|--------------------------------------|--------------|
| Python         | 3.12.6                                         | Runtime                              | ~70 MB      |
| FFmpeg         | latest                                         | Video encoding                       | ~60 MB      |
| MiKTeX         | latest (basic + auto-install packages)         | LaTeX / MathTex rendering            | 200+ MB     |
| Ollama         | latest                                         | Local LLM runner                     | ~100 MB     |
| qwen2.5-coder  | 7b (or 14b if ≥16 GB VRAM)                     | Code generation model                | ~4.7 GB     |

### Step-by-step Installation

1. **Install Python 3.12.6**  
   https://www.python.org/downloads/release/python-3126/  
  

2. **Install FFmpeg**  
   https://ffmpeg.org/download.html  
   → Extract → add the `bin` folder to your system PATH

3. **Install MiKTeX**  
   https://miktex.org/download  
   → It will automatically download missing LaTeX packages on first use

4. **Install & start Ollama**  
   https://ollama.com/download  
   Open a terminal / PowerShell:

```bash
ollama pull qwen2.5-coder:7b

```

In a second terminal window:


```bash
ollama pull qwen2.5-coder:7b

```

# Install dependencies : 

```bash

python -m pip install --upgrade pip
pip install -r requirements.txt

```

# Launch the application :

```bash
streamlit run app.py
```


If you want to see a video of the examples that were created: go to example-result Directory 


# example Use MVP :


https://github.com/user-attachments/assets/962d6e62-e0bd-4986-a3be-2d79f3123613

# Example animation Create : 


https://github.com/user-attachments/assets/3aa0bb32-610e-4796-af96-0b6c70fef764



