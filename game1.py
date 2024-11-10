import cv2
import mediapipe as mp
import pygame
import random
import sys
import queue
from threading import Thread
import logging
import os

# Initialize logging
logging.basicConfig(level=logging.DEBUG)

# Initialize Pygame and its mixer
pygame.init()
pygame.mixer.init()

# Load and play background music
try:
    music_path = r"C:\Users\Asarv\Downloads\41667__universfield__halloween-music\761764__universfield__halloween-mischief-playful-music-for-halloween-festivities.mp3"
    pygame.mixer.music.load(music_path)
    pygame.mixer.music.play(-1)  # -1 means loop indefinitely
    pygame.mixer.music.set_volume(0.5)  # Set volume to 50%
except pygame.error as e:
    logging.error(f"Error loading background music: {e}")

# Constants
WIDTH, HEIGHT = 600, 650
FPS = 60
GRAVITY = 0.5
JUMP_STRENGTH = -10
PIPE_GAP = 150
PIPE_WIDTH = 80

# Initialize Pygame screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Flappy Bird with Hand Tracking")

class Bird:
    def __init__(self):
        try:
            # Load the bird image
            original_image = pygame.image.load(r"C:\Users\Asarv\Documents\Manipal\coding\fcv lab\CV\123.jpeg")
            # Scale image to desired size (adjust size as needed)
            self.image = pygame.transform.scale(original_image, (50, 50))
            logging.info("Bird image loaded successfully")
        except pygame.error as e:
            logging.error(f"Error loading bird image: {e}")
            # Create fallback surface if image loading fails
            self.image = pygame.Surface((50, 50))
            self.image.fill((255, 255, 0))
        
        self.reset()
        
    def reset(self):
        self.rect = self.image.get_rect()
        self.rect.x = 100
        self.rect.y = HEIGHT // 2
        self.velocity = 0

    def jump(self):
        self.velocity = JUMP_STRENGTH

    def update(self):
        self.velocity += GRAVITY
        self.rect.y += self.velocity
        if self.rect.y > HEIGHT - self.rect.height:
            self.rect.y = HEIGHT - self.rect.height
        if self.rect.y < 0:
            self.rect.y = 0

    def draw(self, surface):
        surface.blit(self.image, self.rect)

class Pipe:
    def __init__(self):
        self.reset()

    def reset(self):
        self.height = random.randint(100, 400)
        self.top = self.height - 100
        self.bottom = self.height + PIPE_GAP
        self.x = WIDTH
        self.rect_top = pygame.Rect(self.x, 0, PIPE_WIDTH, self.top)
        self.rect_bottom = pygame.Rect(self.x, self.bottom, PIPE_WIDTH, HEIGHT - self.bottom)

    def update(self):
        self.x -= 5
        self.rect_top.x = self.x
        self.rect_bottom.x = self.x

    def draw(self, surface):
        pygame.draw.rect(surface, (0, 255, 0), self.rect_top)
        pygame.draw.rect(surface, (0, 255, 0), self.rect_bottom)

class VideoGet:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        self.stopped = False
        self.frame_queue = queue.Queue(maxsize=2)
        
    def start(self):
        Thread(target=self.get, daemon=True).start()
        return self
        
    def get(self):
        while not self.stopped:
            if not self.frame_queue.full():
                ret, frame = self.stream.read()
                if ret:
                    self.frame_queue.put(frame)
    
    def read(self):
        return False if self.frame_queue.empty() else self.frame_queue.get()
    
    def stop(self):
        self.stopped = True

class Game:
    def __init__(self):
        self.font = pygame.font.SysFont("Arial", 30)
        self.game_over_font = pygame.font.SysFont("Arial", 50)
        self.game_over_text = self.game_over_font.render("Game Over!", True, (255, 255, 255))
        self.restart_text = self.font.render("Press SPACE to Restart", True, (255, 255, 255))
        
        # Load background image
        try:
            original_bg = pygame.image.load(r"C:\Users\Asarv\Documents\Manipal\coding\fcv lab\CV\images.jpeg")
            self.background = pygame.transform.scale(original_bg, (WIDTH, HEIGHT))
            logging.info("Background image loaded successfully")
        except pygame.error as e:
            logging.error(f"Error loading background image: {e}")
            self.background = pygame.Surface((WIDTH, HEIGHT))
            self.background.fill((0, 0, 255))
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            max_num_hands=1
        )
        
        self.reset()
        
    def reset(self):
        self.bird = Bird()
        self.pipes = [Pipe()]
        self.score = 0
        self.game_over = False
        pygame.mixer.music.play(-1)

    def update(self):
        if not self.game_over:
            self.bird.update()
            if self.bird.rect.y >= HEIGHT - self.bird.rect.height:
                self.game_over = True
                pygame.mixer.music.stop()

            for pipe in self.pipes:
                pipe.update()
                if pipe.x < -PIPE_WIDTH:
                    self.pipes.remove(pipe)
                    self.pipes.append(Pipe())
                    self.score += 1

                if self.bird.rect.colliderect(pipe.rect_top) or self.bird.rect.colliderect(pipe.rect_bottom):
                    self.game_over = True
                    pygame.mixer.music.stop()

    def draw(self):
        # Draw background image instead of fill
        screen.blit(self.background, (0, 0))
        self.bird.draw(screen)
        for pipe in self.pipes:
            pipe.draw(screen)
        score_text = self.font.render(f"Score: {self.score}", True, (255, 255, 255))
        screen.blit(score_text, (10, 10))
        if self.game_over:
            screen.blit(self.game_over_text, (WIDTH // 2 - 100, HEIGHT // 2 - 25))
            screen.blit(self.restart_text, (WIDTH // 2 - 150, HEIGHT // 2 + 25))

def main():
    try:
        game = Game()
        video_getter = VideoGet().start()
        clock = pygame.time.Clock()

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    video_getter.stop()
                    pygame.mixer.music.stop()
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE and game.game_over:
                        game.reset()

            frame = video_getter.read()
            if frame is not False:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = game.hands.process(frame)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        index_finger_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
                        if index_finger_tip.y < 0.5:
                            game.bird.jump()

                game.update()
                game.draw()
                pygame.display.flip()
                clock.tick(FPS)
    except Exception as e:
        logging.error(f"Game error: {e}")
    finally:
        pygame.mixer.music.stop()
        pygame.mixer.quit()
        pygame.quit()

if __name__ == "__main__":
    main()