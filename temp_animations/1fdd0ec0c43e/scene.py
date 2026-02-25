from manim import *
import numpy as np

class MathScene(Scene):
    def construct(self):
        self.camera.background_color = "#0a0a0a"
        
        equation = MathTex("(a+b)^2").set_color(WHITE).scale(3)
        self.play(Write(equation))
        
        self.wait(1)
        
        expanded = MathTex("a^2 + 2ab + b^2").set_color(WHITE).scale(3)
        self.play(Transform(equation, expanded))
        
        self.wait(2)