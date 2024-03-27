import pygame
import math
import colorsys

# Initialize Pygame
pygame.init()

# Set the window size
width, height = 800, 800
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Wheel Game")

# Set the number of slots
M = 12

# Set the center and radius of the wheel
center_x, center_y = width // 2, height // 2
radius = min(width, height) // 2 - 20

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
                angle_slot = -(angle+ 2 * math.pi) / (2 * math.pi / M)
                slot = int(angle_slot) % M
                
                # Get the color of the clicked slot
                hue = slot / M
                color = colorsys.hsv_to_rgb(hue, 1, 1)
                color = tuple(int(c * 255) for c in color)
                
                # Print the result
                print(f"Clicked slot: {slot}, Color: {color}")

    # Clear the screen
    screen.fill((255, 255, 255))

    # Draw the wheel
    for i in range(M):
        angle = i * (2 * math.pi / M)
        hue = i / M
        color = colorsys.hsv_to_rgb(hue, 1, 1)
        color = tuple(int(c * 255) for c in color)
        pygame.draw.arc(screen, color, (20, 20, width - 40, height - 40), angle, angle + (2 * math.pi / M), radius)

    # Update the display
    pygame.display.flip()

# Quit the game
pygame.quit()