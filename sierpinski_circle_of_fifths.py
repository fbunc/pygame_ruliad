from manim import *
import numpy as np

class SierpinskiCircleOfFifths(Scene):
    def construct(self):
        # Constants
        M = 12
        fifths = np.array(['A', 'D', 'G', 'C', 'F', 'Bb', 'Eb', 'Ab', 'Db', 'Gb', 'B', 'E'])
        omega_s = 2 * np.pi / M
        r_min, r_max = 0, 1
        phi_min, phi_max = -np.pi, np.pi
        N_iter = 500

        # Helper function to get the complex number for a note
        def get_note_complex(note):
            k = np.where(fifths == note)[0][0]
            return np.exp(1j * omega_s * k)

        # Create the circle of fifths
        circle = Circle(radius=1, color=WHITE)
        self.play(Create(circle))

        # Add labels for the notes
        for k in range(M):
            note = fifths[k]
            angle = omega_s * k
            label = Text(str(note), font_size=24, color=WHITE).move_to(1.2 * np.array([np.cos(angle), np.sin(angle), 0]))
            self.play(Write(label), run_time=0.2)

        # Randomly choose initial point inside the unitary circle
        r_o = np.random.uniform(r_min, r_max)
        phi_o = np.random.uniform(phi_min, phi_max)
        Z_o = r_o * np.exp(1j * phi_o)

        # Create a dot at the initial point
        dot = Dot(point=[Z_o.real, Z_o.imag, 0], color=RED)
        self.play(Create(dot))

        # Iterate and draw lines
        for _ in range(N_iter):
            # Randomly choose a note from (Ab, E, C)
            note = np.random.choice(['Ab', 'E', 'C'])
            Z_note = get_note_complex(note)

            # Draw a line between Z_o and Z_note
            line = Line(start=[Z_o.real, Z_o.imag, 0], end=[Z_note.real, Z_note.imag, 0], color=RED)
            self.play(Create(line), run_time=0.01)

            # Find the midpoint between Z_o and Z_note
            Z_half = (Z_o + Z_note) / 2

            # Move the dot to the midpoint
            self.play(dot.animate.move_to([Z_half.real, Z_half.imag, 0]), run_time=0.01)

            # Update Z_o with the midpoint
            Z_o = Z_half

        self.wait(2)