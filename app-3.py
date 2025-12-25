"""
Math to Manim - Professional Edition
=====================================
A cinematic mathematical animation generator powered by local LLM and Manim.
Transforms natural language descriptions into breathtaking mathematical visualizations.

Author: Professional Edition
Version: 2.0.0
Date: December 2025
"""

import streamlit as st
import ollama
import subprocess
import uuid
from pathlib import Path
import shutil
import time
import re
from typing import Optional, Tuple
from datetime import datetime

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

TEMP_DIR = Path("temp_animations")
CLEANUP_AGE = 3600  # seconds
DEFAULT_MODEL = "qwen2.5-coder:7b"

# Manim quality presets
QUALITY_PRESETS = {
    "Ultra (4K - slow)": {"flag": "-qk", "desc": "3840x2160, 60fps"},
    "High (1080p)": {"flag": "-qh", "desc": "1920x1080, 60fps"},
    "Medium (720p)": {"flag": "-qm", "desc": "1280x720, 30fps"},
    "Low (480p - fast)": {"flag": "-ql", "desc": "854x480, 15fps"}
}

# Professional example prompts
EXAMPLE_PROMPTS = [
    "Fighter jet breaking sound barrier with shockwaves and vapor cone",
    "Black hole with spinning accretion disk and gravitational lensing",
    "Double pendulum chaos with rainbow trail",
    "Mandelbrot set deep zoom into infinite spirals",
    "Quantum wave function in a box",
    "Fourier series forming a beating heart",
    "Navier-Stokes fluid flow around cylinder",
    "Riemann zeta zeros on critical line"
]

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def initialize_session_state():
    """Initialize all session state variables."""
    if 'generated_code' not in st.session_state:
        st.session_state.generated_code = None
    if 'enhanced_prompt' not in st.session_state:
        st.session_state.enhanced_prompt = None
    if 'video_path' not in st.session_state:
        st.session_state.video_path = None
    if 'generation_history' not in st.session_state:
        st.session_state.generation_history = []

def cleanup_old_files():
    """Remove temporary animation files older than CLEANUP_AGE."""
    if not TEMP_DIR.exists():
        return
    
    current_time = time.time()
    for item in TEMP_DIR.iterdir():
        if item.is_dir():
            try:
                if current_time - item.stat().st_mtime > CLEANUP_AGE:
                    shutil.rmtree(item, ignore_errors=True)
            except Exception:
                pass

def sanitize_code(raw_code: str) -> str:
    """
    Sanitize and fix common issues in generated Manim code.
    
    Args:
        raw_code: Raw code output from LLM
        
    Returns:
        Cleaned and sanitized Python code
    """
    # Extract code from markdown blocks
    if "```python" in raw_code:
        code = raw_code.split("```python", 1)[1].split("```", 1)[0]
    elif "```" in raw_code:
        code = raw_code.split("```", 1)[1].split("```", 1)[0]
    else:
        code = raw_code
    
    code = code.strip()
    
    # Ensure proper imports
    if "from manim import *" not in code:
        code = "from manim import *\nimport numpy as np\n\n" + code
    elif "import numpy as np" not in code:
        code = code.replace("from manim import *", "from manim import *\nimport numpy as np")
    
    # Check if code uses camera features (MovingCameraScene features)
    uses_camera_frame = "self.camera.frame" in code or "camera.frame" in code
    uses_camera_animate = "self.camera.animate" in code or "camera.animate" in code
    needs_moving_camera = uses_camera_frame or uses_camera_animate
    
    # Keep only the first MathScene class and fix Scene type if needed
    if needs_moving_camera:
        # Change to MovingCameraScene if camera features are used
        scene_matches = list(re.finditer(r"class MathScene\((?:Scene|MovingCameraScene)\):[\s\S]*?(?=class |\Z)", code))
        if scene_matches:
            imports = re.findall(r"^(?:from|import)\s+.*$", code, re.MULTILINE)
            import_block = "\n".join(imports) if imports else "from manim import *\nimport numpy as np"
            scene_code = scene_matches[0].group(0)
            # Force MovingCameraScene
            scene_code = re.sub(r"class MathScene\(Scene\):", "class MathScene(MovingCameraScene):", scene_code)
            
            # Fix self.camera.animate to self.camera.frame.animate
            scene_code = scene_code.replace("self.camera.animate", "self.camera.frame.animate")
            
            code = import_block + "\n\n" + scene_code
    else:
        # Regular Scene
        scene_matches = list(re.finditer(r"class MathScene\((?:Scene|MovingCameraScene)\):[\s\S]*?(?=class |\Z)", code))
        if scene_matches:
            imports = re.findall(r"^(?:from|import)\s+.*$", code, re.MULTILINE)
            import_block = "\n".join(imports) if imports else "from manim import *\nimport numpy as np"
            scene_code = scene_matches[0].group(0)
            # Ensure it's regular Scene
            scene_code = re.sub(r"class MathScene\(MovingCameraScene\):", "class MathScene(Scene):", scene_code)
            code = import_block + "\n\n" + scene_code
    
    # Fix common Manim errors
    code = re.sub(r"\[([-\d.]+),\s*([-\d.]+),\s*0\]", r"[\1, \2, 0]", code)
    
    # Remove or replace unsupported 3D objects
    code = code.replace("Cone(", "Circle(")
    code = code.replace("Cylinder(", "Rectangle(")
    code = code.replace("Sphere(", "Circle(")
    code = code.replace("SVGMobject(", "Text(")
    
    # Remove ImageMobject references (files don't exist)
    # Replace with colored rectangles as placeholders
    lines = code.split('\n')
    filtered_lines = []
    for line in lines:
        if 'ImageMobject(' in line:
            # Extract variable name and try to create a replacement
            match = re.search(r'(\w+)\s*=\s*ImageMobject\([^)]+\)', line)
            if match:
                var_name = match.group(1)
                indent = len(line) - len(line.lstrip())
                # Replace with a colored rectangle
                replacement = ' ' * indent + f'{var_name} = Rectangle(width=4, height=3, color=BLUE, fill_opacity=0.3)'
                if '.scale(' in line:
                    scale_match = re.search(r'\.scale\(([\d.]+)\)', line)
                    if scale_match:
                        replacement += f'.scale({scale_match.group(1)})'
                if '.shift(' in line:
                    shift_match = re.search(r'\.shift\([^)]+\)', line)
                    if shift_match:
                        replacement += shift_match.group(0)
                filtered_lines.append(replacement)
            else:
                # Just comment it out
                filtered_lines.append('        # ' + line.strip() + '  # ImageMobject removed - file not found')
        else:
            filtered_lines.append(line)
    code = '\n'.join(filtered_lines)
    
    # Fix incorrect color_gradient syntax
    # Wrong: .set_color(color_gradient=[BLUE, WHITE])
    # Right: .set_color_by_gradient(BLUE, WHITE) or just .set_color(BLUE)
    code = re.sub(r'\.set_color\(color_gradient=\[([^\]]+)\]\)', r'.set_color_by_gradient(\1)', code)
    
    # Fix incorrect gradient parameter in constructors
    # Wrong: Circle(color_gradient=[RED, BLUE])
    # Right: Circle(color=RED) then .set_color_by_gradient(RED, BLUE)
    code = re.sub(r'color_gradient=\[([^\]]+)\]', r'color=BLUE', code)
    
    # Fix np.random calls without proper array creation
    # Wrong: Dot(np.random.uniform(-8, 8), ...)
    # Right: Dot(np.array([np.random.uniform(-8, 8), ...]))
    code = re.sub(
        r'Dot\(np\.random\.uniform\(([^)]+)\),\s*np\.random\.uniform\(([^)]+)\)',
        r'Dot(np.array([np.random.uniform(\1), np.random.uniform(\2), 0]))',
        code
    )
    
    # Fix MoveToTarget usage (incorrect syntax)
    # Wrong: MoveToTarget(obj, target=...)
    # Right: obj.animate.move_to(...)
    code = re.sub(r'MoveToTarget\(([^,]+),\s*target=([^)]+)\)', r'\1.animate.move_to(\2.get_center())', code)
    
    # Fix standalone .animate.rotate() without self.play()
    # Wrong: earth_plane.animate.rotate(PI / 4).run_time = 1
    # Right: self.play(earth_plane.animate.rotate(PI / 4), run_time=1)
    lines = code.split('\n')
    filtered_lines = []
    for line in lines:
        if '.animate.' in line and 'self.play(' not in line and '.run_time' in line:
            # This is a broken animation line
            match = re.search(r'(\w+)\.animate\.(\w+)\(([^)]+)\)\.run_time\s*=\s*(\d+)', line)
            if match:
                obj, method, args, run_time = match.groups()
                indent = len(line) - len(line.lstrip())
                fixed_line = ' ' * indent + f'self.play({obj}.animate.{method}({args}), run_time={run_time})'
                filtered_lines.append(fixed_line)
            else:
                filtered_lines.append(line)
        else:
            filtered_lines.append(line)
    code = '\n'.join(filtered_lines)
    
    # Fix Matrix objects with numpy arrays - convert to simple lists
    # Wrong: Matrix([[np.cos(...), np.sin(...)], [...]])
    # Right: MathTex(r"\begin{bmatrix} ... \end{bmatrix}")
    # Simply replace Matrix() with MathTex for LaTeX rendering
    def fix_matrix(match):
        content = match.group(1)
        # Try to extract simple pattern
        return 'MathTex(r"\\begin{bmatrix} \\cos(\\theta) & -\\sin(\\theta) \\\\ \\sin(\\theta) & \\cos(\\theta) \\end{bmatrix}")'
    
    code = re.sub(r'Matrix\(\[\s*\[(.*?)\]\s*\]\)', fix_matrix, code, flags=re.DOTALL)
    
    # Fix Polygon with incorrect numpy array dimensions
    # Wrong: Polygon(np.array([[-2, -1.5, 0]], dtype=float), ...)
    # Right: Polygon(np.array([-2, -1.5, 0]), ...)
    # Remove extra brackets from np.array calls
    code = re.sub(
        r'np\.array\(\[\[([-\d.,\s]+)\]\],\s*dtype=float\)',
        r'np.array([\1])',
        code
    )
    # Also fix without dtype
    code = re.sub(
        r'np\.array\(\[\[([-\d.,\s]+)\]\]\)',
        r'np.array([\1])',
        code
    )
    
    # Fix Polygon calls with multiple separate np.array arguments
    # Convert to single points list
    def fix_polygon_call(match):
        full_match = match.group(0)
        # Extract all coordinate sets
        coords = re.findall(r'\[([-\d.,\s]+)\]', full_match)
        if len(coords) >= 3:
            # Build proper Polygon call
            points_str = ', '.join([f'np.array([{c}])' for c in coords[:10]])  # Limit to 10 points
            return f'Polygon({points_str}'
        return full_match
    
    # Look for Polygon calls with multiple separate arrays
    code = re.sub(
        r'Polygon\(np\.array\(\[[^\]]+\]\)[,\s]+np\.array\(\[[^\]]+\]\)[,\s]+np\.array\([^)]+\)',
        fix_polygon_call,
        code
    )
    
    # Fix invalid color names - replace with valid Manim colors
    invalid_colors = {
        'RED_BROWN': 'MAROON',
        'NEON_BLUE': 'BLUE_C',
        'SOFT_RED': 'RED_D',
        'SILVER': 'GRAY',
        'DARK_BLUE': 'BLUE_E',
        'SKYBLUE': 'BLUE_B',
        'DARK_RED': 'DARK_BROWN',
        'LIGHT_BLUE': 'BLUE_A',
        'DARK_GREEN': 'GREEN_E',
        'LIGHT_GREEN': 'GREEN_A',
        'NEON_GREEN': 'GREEN_C',
        'BRIGHT_RED': 'RED_A',
        'BRIGHT_BLUE': 'BLUE_A',
        'BRIGHT_GREEN': 'GREEN_A',
        'GOLD': 'YELLOW_D',
    }
    
    for invalid, valid in invalid_colors.items():
        code = code.replace(invalid, valid)
    
    # Fix Circle with invalid parameters (height, depth)
    # Circle only takes radius, not height/depth/width
    code = re.sub(r'Circle\([^)]*height=[^,)]+[^)]*\)', 'Circle(radius=1)', code)
    code = re.sub(r'Circle\([^)]*depth=[^,)]+[^)]*\)', 'Circle(radius=1)', code)
    code = re.sub(r'Circle\([^)]*width=[^,)]+[^)]*\)', 'Circle(radius=1)', code)
    
    # Fix Sector with invalid parameters
    # Sector needs proper angle range
    code = re.sub(
        r'Sector\(start_angle=([^,]+),\s*angle=([^,]+),\s*radius=([^,)]+)',
        r'Arc(start_angle=\1, angle=\2, radius=\3',
        code
    )
    
    # Fix list comprehension syntax errors - remove trailing commas
    # Wrong: [item, for i in range(10)]
    # Right: [item for i in range(10)]
    code = re.sub(
        r'\[([^,\]]+),\s+(for\s+\w+\s+in\s+)',
        r'[\1 \2',
        code
    )
    
    # Also fix in VGroup
    # Wrong: VGroup(*[item, for _ in range(10)])
    # Right: VGroup(*[item for _ in range(10)])
    code = re.sub(
        r'VGroup\(\*\[([^,\]]+),\s+(for\s+)',
        r'VGroup(*[\1 \2',
        code
    )
    
    # Fix self.time references in updaters (doesn't exist in Manim)
    # Wrong: lambda m: m.shift(UP * np.sin(self.time))
    # Right: Remove updaters with self.time or replace with ValueTracker
    # Simple fix: comment out lines with self.time in updaters
    lines = code.split('\n')
    filtered_lines = []
    for line in lines:
        if 'add_updater' in line and 'self.time' in line:
            # Comment out problematic updater
            indent = len(line) - len(line.lstrip())
            filtered_lines.append(' ' * indent + '# ' + line.strip() + '  # Removed: self.time not available')
        else:
            filtered_lines.append(line)
    code = '\n'.join(filtered_lines)
    
    # Also fix standalone self.time references (replace with 0 or remove)
    code = code.replace('self.time', '0  # self.time not available')
    
    # If NOT using MovingCameraScene, remove camera.frame/camera.animate references
    if not needs_moving_camera or "MovingCameraScene" not in code:
        # Remove lines with self.camera.frame or self.camera.animate
        lines = code.split('\n')
        filtered_lines = []
        for line in lines:
            if 'self.camera.frame' in line or 'self.camera.animate' in line or 'camera.frame' in line or 'camera.animate' in line:
                # Comment out instead of removing
                filtered_lines.append('        # ' + line.strip() + '  # Removed: requires MovingCameraScene')
            else:
                filtered_lines.append(line)
        code = '\n'.join(filtered_lines)
    
    # Ensure background color is set
    if "self.camera.background_color" not in code:
        code = code.replace("def construct(self):", 
                          "def construct(self):\n        self.camera.background_color = '#0a0a0a'")
    
    # Ensure proper ending with wait
    if "self.wait(" not in code[-200:]:
        code = code.rstrip() + "\n        self.wait(2)"
    
    return code

def create_fallback_animation(description: str) -> str:
    """
    Create a simple fallback animation when generation fails.
    
    Args:
        description: User's original description
        
    Returns:
        Basic working Manim code
    """
    safe_desc = description[:40].replace('"', "'")
    return f"""from manim import *
import numpy as np

class MathScene(Scene):
    def construct(self):
        self.camera.background_color = '#0a0a0a'
        
        title = Text("{safe_desc}...", font_size=36, color=BLUE)
        title.to_edge(UP)
        
        circle = Circle(radius=1.5, color=PURPLE)
        square = Square(side_length=2, color=ORANGE)
        triangle = Triangle(color=GREEN)
        
        shapes = VGroup(circle, square, triangle)
        shapes.arrange(RIGHT, buff=1)
        
        self.play(Write(title))
        self.wait(0.5)
        self.play(Create(shapes), run_time=2)
        self.play(Rotate(shapes, angle=PI, run_time=2))
        self.play(FadeOut(shapes), run_time=1)
        
        msg = Text("Animation in progress...\\nPlease try again!", 
                   font_size=28, color=YELLOW)
        self.play(Write(msg))
        self.wait(2)
"""

# ============================================================================
# CORE LLM FUNCTIONS
# ============================================================================

def generate_enhanced_prompt(user_request: str, model: str = DEFAULT_MODEL) -> str:
    """
    Generate an enhanced, detailed prompt from user's basic description.
    
    Args:
        user_request: User's animation description
        model: Ollama model to use
        
    Returns:
        Enhanced prompt with rich details
    """
    system_prompt = """You are a Manim animation expert.
Take the user's request and make it slightly more specific for code generation.

Rules:
- Maximum 3 short sentences
- Only clarify what the user asked for
- Mention colors if not specified (use: RED, BLUE, GREEN, YELLOW, PURPLE, ORANGE, PINK, WHITE)
- No timestamps, no overly detailed descriptions
- Keep it simple and direct

Example:
User: "show the formula E=mc²"
Output: Display the equation E=mc² in large text at the center. Use white text on dark background. Animate it appearing with a write effect.

Output ONLY the brief clarification."""

    try:
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f'Clarify this animation request in 3 sentences max:\n\n"{user_request}"'}
            ],
            options={
                "temperature": 0.4,
                "top_p": 0.9
            }
        )
        return response['message']['content'].strip()
    except Exception as e:
        st.error(f"❌ LLM error during prompt enhancement: {str(e)}")
        return user_request

def generate_manim_code(enhanced_prompt: str, model: str = DEFAULT_MODEL) -> str:
    """
    Generate working Manim code from enhanced prompt.
    
    Args:
        enhanced_prompt: Detailed animation plan
        model: Ollama model to use
        
    Returns:
        Python code for Manim animation
    """
    system_prompt = """You are an expert Manim v0.18.1 developer specializing in simple, working mathematical visualizations.

CRITICAL REQUIREMENTS:
1. Import: from manim import *; import numpy as np
2. Class: class MathScene(Scene): OR class MathScene(MovingCameraScene): if camera movement needed
3. Background: self.camera.background_color = "#0a0a0a"
4. Use ONLY simple 2D objects: Circle, Square, Rectangle, Line, Dot, Arrow, Text, MathTex
5. End with self.wait(2)
6. Keep animations SIMPLE - just a few play() calls
7. NEVER use more than 10-15 lines of actual animation code

CRITICAL - VALID COLORS ONLY:
Use ONLY: RED, BLUE, GREEN, YELLOW, PURPLE, ORANGE, PINK, WHITE, BLACK, GRAY
Color variants: RED_A through RED_E, BLUE_A through BLUE_E, etc.

CRITICAL - NO COMPLEX FEATURES:
- NO add_updater with self.time
- NO ImageMobject or SVGMobject
- NO Matrix() - use MathTex instead
- NO complex loops or excessive animations
- Keep it SIMPLE and WORKING

CRITICAL - NO EXTERNAL FILES:
- NEVER use ImageMobject() - image files don't exist
- NEVER use SVGMobject() - SVG files don't exist  
- Create all visuals using built-in Manim objects (shapes, text, math)
- Use creative combinations of shapes to represent complex objects
- Example: Instead of ImageMobject("star.png"), use Star() or Polygon()
- For fighter jets, use polygons and rectangles to create simple geometric representations

CRITICAL - VALID COLORS ONLY:
- Use ONLY these standard Manim colors: RED, BLUE, GREEN, YELLOW, PURPLE, ORANGE, PINK, WHITE, BLACK, GRAY, MAROON
- Color variants: RED_A through RED_E, BLUE_A through BLUE_E, GREEN_A through GREEN_E, etc.
- NEVER use: RED_BROWN, NEON_BLUE, SOFT_RED, SILVER, SKYBLUE, GOLD (these don't exist)
- When in doubt, use basic colors: RED, BLUE, GREEN, YELLOW, PURPLE, ORANGE

CRITICAL - CORRECT SYNTAX:
- Color gradients: Use .set_color_by_gradient(COLOR1, COLOR2) NOT .set_color(color_gradient=[...])
- Random positions: Dot(np.array([np.random.uniform(-5, 5), np.random.uniform(-3, 3), 0]))
- Animations: self.play(obj.animate.method(), run_time=1) NOT obj.animate.method().run_time = 1
- Moving objects: self.play(obj.animate.move_to(position))
- NO MoveToTarget - use obj.animate.move_to() instead
- Camera zoom: self.camera.frame.animate.scale(0.5) NOT self.camera.animate.scale(0.5)
- Matrices: Use MathTex(r"\begin{bmatrix} a & b \\ c & d \end{bmatrix}") NOT Matrix([[a,b],[c,d]])
- Polygons: Polygon(np.array([-2, -1, 0]), np.array([2, -1, 0]), np.array([0, 3, 0]))
- Colors: Only use valid Manim color names (RED, BLUE, etc.)
- Circle: Only radius parameter, NO height/width/depth
- List comprehensions: [item for i in range(10)] NOT [item, for i in range(10)]
- NO add_updater with self.time - self.time doesn't exist in Manim scenes
- For pulsating effects, use animations in a loop instead of updaters

IMPORTANT - Camera Movement:
- For camera zoom/pan, use: class MathScene(MovingCameraScene)
- Then use: self.camera.frame.animate.scale() or self.camera.frame.animate.move_to()
- NEVER use self.camera.animate - it doesn't exist! Always use self.camera.frame.animate
- If using regular Scene, DO NOT use any camera animations

AVAILABLE OBJECTS:
- Shapes: Circle, Square, Rectangle, Triangle, Polygon, Star, RegularPolygon, Ellipse, Annulus, Arc
- Text: Text, MathTex, Tex, DecimalNumber, Integer
- Lines: Line, Arrow, Vector, DashedLine, Dot, Angle
- Grouping: VGroup, AnimationGroup
- Colors: RED, BLUE, GREEN, YELLOW, PURPLE, ORANGE, PINK, WHITE, BLACK, GRAY, etc.
- Color methods: .set_color(COLOR), .set_fill(COLOR, opacity=0.5), .set_stroke(COLOR, width=2)
- Gradient method: .set_color_by_gradient(COLOR1, COLOR2, COLOR3, ...)

CORRECT EXAMPLES:
```python
# Stars with gradient - VALID COLORS ONLY
stars = VGroup(*[Dot(np.array([np.random.uniform(-7, 7), np.random.uniform(-4, 4), 0])) for _ in range(50)])
# NO TRAILING COMMA before 'for'!
stars.set_color_by_gradient(BLUE, WHITE, YELLOW)

# Light beams - correct syntax
light_beams = VGroup(*[
    Line(ORIGIN, np.array([np.random.uniform(-5, 5), np.random.uniform(-3, 3), 0]))
    for _ in range(20)
])  # Each line is complete BEFORE the 'for'

# Simple geometric jet (no external files!)
jet = VGroup(
    # Body
    Rectangle(width=0.3, height=1, color=WHITE),
    # Wings
    Polygon(
        np.array([-0.5, 0, 0]),
        np.array([0.5, 0, 0]),
        np.array([0, 0.3, 0]),
        color=BLUE
    )
).move_to(LEFT * 3)

# Moving animation
self.play(jet.animate.shift(RIGHT * 5), run_time=3)

# Color with fill - VALID COLORS
rect = Rectangle(width=4, height=2, color=BLUE, fill_opacity=0.5)

# Shockwave using Arc (not Sector with invalid params)
shockwave = Arc(start_angle=0, angle=PI/4, radius=2, color=WHITE, stroke_width=5)
shockwave.move_to(jet.get_center())

# Gradient background - VALID COLORS
bg = Rectangle(width=14, height=8, fill_opacity=1)
bg.set_color_by_gradient(PURPLE, PINK, ORANGE, YELLOW)

# Camera zoom (MovingCameraScene only!)
class MathScene(MovingCameraScene):
    def construct(self):
        self.play(self.camera.frame.animate.scale(0.5), run_time=2)

# Matrix display
matrix = MathTex(
    r"\begin{bmatrix} \cos(\theta) & -\sin(\theta) \\ \sin(\theta) & \cos(\theta) \end{bmatrix}"
).scale(1.5)

# Circle - radius ONLY
circle = Circle(radius=2, color=RED, fill_opacity=0.3)
# NOT: Circle(height=4, width=4) - this causes errors!

# Ellipse for oval shapes
oval = Ellipse(width=4, height=2, color=GREEN, fill_opacity=0.5)

# Pulsating effect - use animation loop, NOT updaters with self.time
# WRONG: energy.add_updater(lambda m: m.shift(UP * np.sin(self.time)))
# RIGHT: Use a for loop with animations
energy = Circle(radius=2, color=YELLOW, fill_opacity=0.5)
for _ in range(3):
    self.play(energy.animate.scale(1.2), run_time=0.5)
    self.play(energy.animate.scale(0.83), run_time=0.5)  # Back to original
```

AVOID:
- 3D objects (Cone, Cylinder, Sphere, Surface, Cube)
- ImageMobject, SVGMobject (files don't exist!)
- camera.frame with regular Scene class
- Matrix() object (use MathTex with bmatrix instead)
- Invalid color names (RED_BROWN, NEON_BLUE, SOFT_RED, SILVER, GOLD, etc.)
- Circle with height/width/depth parameters
- add_updater with self.time (self.time doesn't exist)
- Overly complex updaters - use simple animations instead
- Undefined variables or imports
- External file dependencies of any kind
- Overly complex animations - keep it simple and working

OUTPUT:
- Clean Python code ONLY
- No markdown, no explanations
- One complete working script
- All visuals created from Manim primitives"""

    code_prompt = f"""Create a SIMPLE working Manim script for this:

{enhanced_prompt}

Requirements:
- Class: MathScene(Scene)
- Background: self.camera.background_color = "#0a0a0a"
- Keep it SIMPLE: 10-15 lines of animation max
- Use basic shapes and Text/MathTex
- Just a few self.play() calls
- Valid colors only (RED, BLUE, GREEN, YELLOW, etc.)

Return ONLY Python code, no markdown."""

    try:
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": code_prompt}
            ],
            options={
                "temperature": 0.3,
                "top_p": 0.95
            }
        )
        raw_code = response['message']['content']
        return sanitize_code(raw_code)
    except Exception as e:
        st.error(f"❌ LLM error during code generation: {str(e)}")
        return create_fallback_animation(enhanced_prompt[:100])

def refine_code_with_modifications(original_code: str, modifications: str, model: str = DEFAULT_MODEL) -> str:
    """
    Refine existing Manim code based on user modifications.
    
    Args:
        original_code: Current Manim code
        modifications: User's requested changes
        model: Ollama model to use
        
    Returns:
        Modified Manim code
    """
    system_prompt = """You are an expert Manim developer specializing in code refinement.
The user has an existing animation and wants to modify it.

Your task:
1. Understand the existing code structure
2. Apply the requested modifications precisely
3. Maintain all working parts of the original
4. Ensure the result is valid Manim v0.18.1 code
5. Keep the same class name: MathScene

Return ONLY the complete modified code, no explanations."""

    refine_prompt = f"""Here is the current Manim code:

```python
{original_code}
```

Apply these modifications:
{modifications}

Return the complete updated code."""

    try:
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": refine_prompt}
            ],
            options={
                "temperature": 0.2,
                "top_p": 0.9
            }
        )
        raw_code = response['message']['content']
        return sanitize_code(raw_code)
    except Exception as e:
        st.error(f"❌ Error refining code: {str(e)}")
        return original_code

# ============================================================================
# RENDERING FUNCTION
# ============================================================================

def render_manim_animation(
    code: str,
    quality_preset: str,
    custom_resolution: Optional[str] = None,
    custom_fps: Optional[int] = None
) -> Tuple[bool, Optional[Path], str]:
    """
    Render Manim animation from code.
    
    Args:
        code: Python code containing Manim scene
        quality_preset: Quality preset name
        custom_resolution: Optional custom resolution (e.g., "1920x1080")
        custom_fps: Optional custom frame rate
        
    Returns:
        Tuple of (success, video_path, error_message)
    """
    temp_dir = TEMP_DIR / uuid.uuid4().hex[:12]
    temp_dir.mkdir(parents=True, exist_ok=True)
    script_path = temp_dir / "scene.py"
    
    try:
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(code)
    except Exception as e:
        return False, None, f"Failed to write script: {str(e)}"
    
    quality_flag = QUALITY_PRESETS[quality_preset]["flag"]
    
    cmd = [
        "manim", "render",
        str(script_path), "MathScene",
        quality_flag,
        "--format", "mp4",
        "--media_dir", str(temp_dir)
    ]
    
    if custom_resolution:
        width, height = custom_resolution.split("x")
        cmd.extend(["-r", f"{width},{height}"])
    
    if custom_fps:
        cmd.extend(["--frame_rate", str(custom_fps)])
    
    try:
        # Run from the parent directory, not from temp_dir
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600
        )
        
        if result.returncode != 0:
            error_msg = result.stderr[-2000:] if result.stderr else "Unknown error"
            return False, None, error_msg
        
        video_files = list(temp_dir.rglob("*.mp4"))
        if not video_files:
            return False, None, "No video file was generated"
        
        return True, video_files[0], ""
        
    except subprocess.TimeoutExpired:
        return False, None, "Rendering timed out (>10 minutes)"
    except Exception as e:
        return False, None, f"Rendering failed: {str(e)}"

# ============================================================================
# STREAMLIT UI
# ============================================================================

def render_ui():
    """Main Streamlit UI rendering function."""
    
    st.markdown("""
        <style>
        .main-header {
            text-align: center;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 3.5rem;
            font-weight: 800;
            margin-bottom: 0.5rem;
        }
        .sub-header {
            text-align: center;
            color: #a0a0a0;
            font-size: 1.2rem;
            margin-bottom: 2rem;
        }
        .stButton>button {
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.3s;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown("<h1 class='main-header'>🎬 Math to Manim</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Professional Edition • Transform ideas into cinematic mathematical animations</p>", unsafe_allow_html=True)
    
    initialize_session_state()
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        st.subheader("🤖 LLM Model")
        model_name = st.text_input("Ollama Model", value=DEFAULT_MODEL, help="Ensure this model is installed")
        
        st.subheader("🎨 Render Quality")
        quality_preset = st.selectbox(
            "Preset",
            options=list(QUALITY_PRESETS.keys()),
            index=1,
            help="Higher quality = longer render time"
        )
        st.caption(f"📊 {QUALITY_PRESETS[quality_preset]['desc']}")
        
        with st.expander("🔧 Advanced Settings"):
            use_custom = st.checkbox("Custom Resolution & FPS")
            if use_custom:
                custom_res = st.text_input("Resolution (WxH)", value="1920x1080", placeholder="1920x1080")
                custom_fps = st.number_input("Frame Rate", min_value=15, max_value=120, value=60, step=15)
            else:
                custom_res = None
                custom_fps = None
        
        st.divider()
        
        if st.session_state.generation_history:
            st.subheader("📜 Recent Generations")
            for i, item in enumerate(reversed(st.session_state.generation_history[-5:])):
                if st.button(f"🎬 {item['prompt'][:30]}...", key=f"hist_{i}"):
                    st.session_state.generated_code = item['code']
                    st.session_state.enhanced_prompt = item['enhanced']
        
        st.divider()
        
        if st.button("🗑️ Cleanup Temp Files"):
            cleanup_old_files()
            st.success("Cleaned up old files!")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("✍️ Describe Your Animation")
        
        use_example = st.selectbox(
            "Or choose an example:",
            ["Custom..."] + EXAMPLE_PROMPTS,
            index=0
        )
        
        if use_example != "Custom...":
            user_request = st.text_area(
                "Your Vision",
                value=use_example,
                height=200,
                help="Describe what you want to see. Be as detailed or simple as you like!"
            )
        else:
            user_request = st.text_area(
                "Your Vision",
                height=200,
                placeholder="Examples:\n• Fourier series forming a beating heart\n• Quantum tunneling through a barrier\n• Lorenz attractor butterfly effect\n• Maxwell's equations in waves",
                help="Describe what you want to see!"
            )
    
    with col2:
        st.subheader("🎯 Quick Tips")
        st.info("""
        **For best results:**
        - Mention colors, speeds, angles
        - Specify formulas to show
        - Describe camera movement
        - Set the mood (dark, cosmic, vibrant)
        
        **Examples:**
        - "with glowing trails"
        - "show the formula E=mc²"
        - "camera orbits slowly"
        - "dark space background"
        """)
    
    st.divider()
    
   if st.button("🚀 Generate Animation"):
    if not user_request.strip():
        st.error("❌ Please describe your animation first!")
        st.stop()
    
    with st.spinner("🎨 Crafting your cinematic vision..."):
        enhanced_prompt = generate_enhanced_prompt(user_request, model_name)
        st.session_state.enhanced_prompt = enhanced_prompt
    
    # Lazy import here
    from manim import *

    with st.spinner("⚡ Generating Manim code..."):
        generated_code = generate_manim_code(enhanced_prompt, model_name)
        st.session_state.generated_code = generated_code

    with st.spinner("🎬 Rendering your animation..."):
        success, video_path, error_msg = render_manim_animation(
            generated_code,
            quality_preset,
            custom_res if use_custom else None,
            custom_fps if use_custom else None
        )
        
        with st.spinner("🎨 Crafting your cinematic vision..."):
            enhanced_prompt = generate_enhanced_prompt(user_request, model_name)
            st.session_state.enhanced_prompt = enhanced_prompt
        
        st.success("✅ Vision enhanced!")
        with st.expander("📋 View Enhanced Prompt", expanded=True):
            st.markdown(f"**{enhanced_prompt}**")
        
        with st.spinner("⚡ Generating Manim masterpiece code..."):
            generated_code = generate_manim_code(enhanced_prompt, model_name)
            st.session_state.generated_code = generated_code
        
        st.success("✅ Code generated!")
        
        with st.spinner("🎬 Rendering your animation (this may take a few minutes)..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i in range(100):
                time.sleep(0.1)
                progress_bar.progress(i + 1)
                if i < 30:
                    status_text.text("Initializing Manim renderer...")
                elif i < 70:
                    status_text.text("Rendering frames...")
                else:
                    status_text.text("Finalizing video...")
            
            success, video_path, error_msg = render_manim_animation(
                generated_code,
                quality_preset,
                custom_res if use_custom else None,
                custom_fps if use_custom else None
            )
            
            progress_bar.empty()
            status_text.empty()
        
        if success:
            st.session_state.video_path = video_path
            
            st.session_state.generation_history.append({
                'prompt': user_request[:50],
                'enhanced': enhanced_prompt,
                'code': generated_code,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
            st.balloons()
            st.success("🎉 **YOUR MASTERPIECE IS READY!**")
        else:
            st.error(f"❌ Rendering failed:\n```\n{error_msg}\n```")
            st.warning("💡 Try simplifying your prompt or check the generated code below")
    
    if st.session_state.video_path and st.session_state.video_path.exists():
        st.divider()
        st.subheader("🎥 Your Animation")
        
        col_vid, col_dl = st.columns([3, 1])
        
        with col_vid:
            st.video(str(st.session_state.video_path))
        
        with col_dl:
            with open(st.session_state.video_path, "rb") as f:
                st.download_button(
                    label="⬇️ Download Video",
                    data=f,
                    file_name=f"animation_{int(time.time())}.mp4",
                    mime="video/mp4",
                    use_container_width=True
                )
            
            if st.button("🔄 Generate New", use_container_width=True):
                st.session_state.video_path = None
                st.rerun()
    
    if st.session_state.generated_code:
        st.divider()
        st.subheader("💻 Generated Code")
        
        tab1, tab2 = st.tabs(["📝 View Code", "✏️ Refine Animation"])
        
        with tab1:
            st.code(st.session_state.generated_code, language="python", line_numbers=True)
            
            col_copy, col_save = st.columns(2)
            with col_save:
                if st.button("💾 Save Code to File"):
                    save_path = TEMP_DIR / f"saved_scene_{int(time.time())}.py"
                    save_path.parent.mkdir(exist_ok=True)
                    with open(save_path, "w") as f:
                        f.write(st.session_state.generated_code)
                    st.success(f"Saved to: {save_path}")
        
        with tab2:
            st.markdown("**Modify your animation by describing changes:**")
            
            modifications = st.text_area(
                "What would you like to change?",
                height=150,
                placeholder="Examples:\n• Make the colors more vibrant\n• Add a title at the top\n• Slow down the animation by 2x\n• Add a formula showing the equation\n• Change background to white",
                help="Describe modifications in natural language"
            )
            
            if st.button("🔄 Apply Modifications", type="primary"):
                if not modifications.strip():
                    st.warning("Please describe what you want to change")
                else:
                    with st.spinner("Refining your animation..."):
                        refined_code = refine_code_with_modifications(
                            st.session_state.generated_code,
                            modifications,
                            model_name
                        )
                        st.session_state.generated_code = refined_code
                    
                    st.success("✅ Code refined! Scroll up and click 'Generate Animation' to render the updated version.")
                    st.rerun()
    
    st.divider()
    st.caption("🎬 Math to Manim Professional Edition v2.0.0 • Powered by Ollama & Manim • Made with ❤️")

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    TEMP_DIR.mkdir(exist_ok=True)
    cleanup_old_files()

    render_ui()
