import pygame
import math
import colorsys

# Initialize Pygame
pygame.init()

# Set the window size
width, height = 800, 800
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Wheel Game")

# Set the maximum number of slots
MAX_M = 64

# Set the initial number of slots
M = 12

# Set the center and radius of the wheel
center_x, center_y = width // 2, height // 2
radius = min(width, height) // 2 - 50

# Create a dropdown menu for selecting M
dropdown_font = pygame.font.Font(None, 24)

# Colors
white = (255, 255, 255)
black = (0, 0, 0)
gray = (200, 200, 200)

# Main game loop
running = True
while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            # Check if the click is inside the wheel
            mouse_x, mouse_y = pygame.mouse.get_pos()
            distance = math.sqrt((mouse_x - center_x) ** 2 + (mouse_y - center_y) ** 2)
            if distance <= radius:
                # Calculate the angle and determine the clicked slot
                angle = math.atan2(mouse_y - center_y, mouse_x - center_x)
                angle_slot = -(angle + 2 * math.pi) / (2 * math.pi / M)
                slot = int(angle_slot) % M

                # Get the color of the clicked slot
                hue = slot / M
                color = colorsys.hsv_to_rgb(hue, 1, 1)
                color = tuple(int(c * 255) for c in color)

                # Print the result
                print(f"Clicked slot: {slot}, Color: {color}")
            # Check if the click is on the dropdown menu
            elif mouse_x >= 20 and mouse_x <= 120 and mouse_y >= 20 and mouse_y <= 40:
                if event.button == 1:  # Left click
                    M = max(2, M - 1)
                elif event.button == 3:  # Right click
                    M = min(MAX_M, M + 1)

    # Clear the screen
    screen.fill(black)

    # Draw the wheel
    for i in range(M):
        angle = i * (2 * math.pi / M)
        hue = i / M
        color = colorsys.hsv_to_rgb(hue, 1, 1)
        color = tuple(int(c * 255) for c in color)
        pygame.draw.arc(screen, color, (50, 50, width - 100, height - 100), angle, angle + (2 * math.pi / M), radius)

        # Draw the label (L_i) outside the circle in a counterclockwise order
        label = f"L_{M-i-1}"
        label_font = pygame.font.Font(None, 24)
        label_text = label_font.render(label, True, gray)
        label_angle = angle + (2 * math.pi / M) * 1.5
        label_x = center_x + (radius + 20) * math.cos(label_angle)
        label_y = center_y + (radius + 20) * math.sin(label_angle)
        label_rect = label_text.get_rect(center=(label_x, label_y))
        screen.blit(label_text, label_rect)

    # Draw the dropdown menu
    dropdown_rect = pygame.Rect(20, 20, 100, 20)
    pygame.draw.rect(screen, gray, dropdown_rect)
    dropdown_text = dropdown_font.render(str(M), True, black)
    dropdown_text_rect = dropdown_text.get_rect(center=dropdown_rect.center)
    screen.blit(dropdown_text, dropdown_text_rect)

    # Update the display
    pygame.display.flip()

# Quit the game
pygame.quit()