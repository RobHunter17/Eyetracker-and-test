import pygame
from sys import exit
import random
from socket import timeout
import constants


pygame.init()
#screen = pygame.display.set_mode((width,height))
screen = pygame.display.set_mode((constants.DISPSIZE))
#screen.fill(125,125,125,255)
pygame.display.set_caption('Eye Tracker')
clock = pygame.time.Clock()
test_font = pygame.font.Font(None, 24)


test_surface = pygame.Surface((constants.DISPSIZE))
test_surface.fill('Grey40')
#text_surface = test_font.render(text, AA,color)
text_surface = test_font.render('When you see a cross, look at it and press s. Then make an eye movement to the black circle when it appears. (press s to start)', True, 'Black')

test_surface2 = pygame.image.load('bc.png').convert_alpha()
circle_x_pos = int(constants.DISPSIZE[0]*0.25)
DEFAULT_IMAGE_SIZE = (200,200)
#pygame.transform.scale('bc.png', DEFAULT_IMAGE_SIZE)

#timer variables
current_time = 0
button_press_time = 0

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()
        if event.type == pygame.KEYDOWN:
            button_press_time = pygame.time.get_ticks()
            

    current_time = pygame.time.get_ticks()
    print(current_time)

    print(f"current time: {current_time} button press time: {button_press_time}")

    screen.blit(test_surface,(0,0))
    
    screen.blit(text_surface,(150,50))
    circle_x_pos -= 3
    if circle_x_pos < -100: circle_x_pos = 800
    screen.blit(test_surface2,(circle_x_pos,250))

    pygame.display.update()
    clock.tick(60)

