import cv2
import numpy as np
import pygame
import random
import threading
import logging
import sys
import mediapipe as mp

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('game_debug.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

class GameException(Exception):
    """Custom exception class for game-specific errors"""
    pass

class EndlessRunner:
    def __init__(self):
        try:
            # Initialize video capture
            self.init_camera()
            
            # Initialize Pygame
            pygame.init()
            if not pygame.get_init():
                raise GameException("Failed to initialize pygame")

            self.game_width, self.game_height = 800, 600
            self.game_screen = pygame.display.set_mode((self.game_width, self.game_height))
            pygame.display.set_caption("Endless Runner")

            # Colors and game states
            self.init_colors()
            self.game_state = "running"  # States: "running", "paused", "game_over"
            
            # Initialize game variables
            self.init_game_variables()
            
            logging.info("Game initialized successfully")
            
        except Exception as e:
            logging.error(f"Initialization error: {str(e)}")
            self.cleanup()
            raise

    def init_camera(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise GameException("Failed to open camera")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)  # Set FPS explicitly
        
        # Initialize MediaPipe for hand tracking
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
        
    def init_colors(self):
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.bg_colors = [
            (135, 206, 235),  # Sky blue
            (255, 182, 193),  # Light pink
            (152, 251, 152),  # Pale green
            (221, 160, 221)   # Plum
        ]
        self.current_bg = 0
        self.bg_transition = 0

    def init_game_variables(self):
        self.player_width, self.player_height = 50, 50
        self.player_x = self.game_width // 4
        self.player_y = self.game_height // 2
        self.player_speed = 5
        self.obstacle_width, self.obstacle_height = 50, 50
        self.obstacle_speed = 5
        self.obstacles = []
        self.score = 0
        self.font = pygame.font.Font(None, 36)
        self.running = True
        self.finger_y = self.game_height // 2
        self.last_finger_y = self.game_height // 2
        self.smoothing_factor = 0.7
        self.fps_counter = 0
        self.fps_timer = pygame.time.get_ticks()
        self.last_frame_time = pygame.time .get_ticks()

    def smooth_movement(self, new_y):
        """Apply smoothing to finger movement"""
        smoothed_y = (self.last_finger_y * self.smoothing_factor + 
                     new_y * (1 - self.smoothing_factor))
        self.last_finger_y = smoothed_y
        return int(smoothed_y)

    def detect_finger_position(self, frame):
        """Detect finger position using MediaPipe"""
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Corrected line
            results = self.hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Get index finger tip coordinates
                    index_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    y = int(index_finger_tip.y * self.game_height)
                    return self.smooth_movement(y)

            return self.finger_y  # Return previous position if no hand detected
            
        except Exception as e:
            logging.error(f"Finger detection error: {str(e)}")
            return self.finger_y

    def update_background(self):
        try:
            self.bg_transition += 0.01
            if self.bg_transition >= 1:
                self.bg_transition = 0
                self.current_bg = (self.current_bg + 1) % len(self.bg_colors)
                
            current_color = self.bg_colors[self.current_bg]
            next_color = self.bg_colors[(self.current_bg + 1) % len(self.bg_colors)]
            
            interpolated_color = tuple(
                int(current_color[i] * (1 - self.bg_transition) + 
                    next_color[i] * self.bg_transition)
                for i in range(3)
            )
            return interpolated_color
        except Exception as e:
            logging.error(f"Background update error: {str(e)}")
            return self.bg_colors[0]

    def process_video(self):
        try:
            last_process_time = pygame.time.get_ticks()
            while self.running:
                current_time = pygame.time.get_ticks()
                # Limit processing to every 16ms (approximately 60 FPS)
                if current_time - last_process_time < 16:
                    continue
                
                ret, frame = self.cap.read()
                if not ret:
                    logging.warning("Failed to read frame from camera")
                    continue

                frame = cv2.flip(frame, 1)
                self.finger_y = self.detect_finger_position(frame)

                # Debug visualization
                cv2.imshow('Debug View', frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.running = False
                elif key == ord('r') and self.game_state == "game_over":
                    self.restart_game()

                last_process_time = current_time
        except Exception as e:
            logging.error(f"Video processing error: {str(e)}")
            self.running = False

    def restart_game(self):
        """Reset game variables for a new game"""
        self.init_game_variables()
        self.game_state = "running"
        logging.info("Game restarted")

    def run_game(self):
        try:
            clock = pygame.time.Clock()

            while self.running:
                current_time = pygame.time.get_ticks()
                delta_time = (current_time - self.last_frame_time) / 1000.0  # Convert to seconds
                self.last_frame_time = current_time

                # Handle events
                self.handle_events()

                if self.game_state == "running":
                    self.update_game(delta_time)
                    self.draw_game()
                elif self.game_state == "game_over":
                    self.draw_game_over()

                # FPS monitoring
                self.monitor_fps()
                clock.tick(60)

        except Exception as e:
            logging.error(f"Game runtime error: {str(e)}")
            self.running = False

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r and self.game_state == "game_over":
                    self.restart_game()
                elif event.key == pygame.K_ESCAPE:
                    self.running = False

    def update_game(self, delta_time):
        try:
            # Update player position with bounds checking
            self.player_y = min(max(self.finger_y, self.player_height), 
                              self.game_height - self.player_height)

            # Update obstacles
            for obstacle in self.obstacles:
                obstacle.x -= self.obstacle_speed * delta_time * 60  # Scale with delta time

            # Spawn new obstacles
            if random.random() < 0.02:  # Adjusted spawn rate
                self.obstacles .append(
                    pygame.Rect(self.game_width, 
                              random.randint(0, self.game_height - self.obstacle_height),
                              self.obstacle_width, 
                              self.obstacle_height)
                )

            # Remove off-screen obstacles
            self.obstacles = [obs for obs in self.obstacles if obs.right > 0]

            # Check collisions
            player_rect = pygame.Rect(self.player_x, self.player_y, 
                                    self.player_width, self.player_height)
            for obstacle in self.obstacles:
                if player_rect.colliderect(obstacle):
                    self.game_state = "game_over"
                    return

            self.score += 1

        except Exception as e:
            logging.error(f"Game update error: {str(e)}")
            self.game_state = "game_over"

    def draw_game(self):
        try:
            # Draw background
            bg_color = self.update_background()
            self.game_screen.fill(bg_color)

            # Draw player
            pygame.draw.rect(self.game_screen, self.BLACK, 
                           (self.player_x, self.player_y, 
                            self.player_width, self.player_height))

            # Draw obstacles
            for obstacle in self.obstacles:
                pygame.draw.rect(self.game_screen, self.RED, obstacle)

            # Draw score
            score_text = self.font.render(f"Score: {self.score}", True, self.BLACK)
            self.game_screen.blit(score_text, (10, 10))

            pygame.display.flip()

        except Exception as e:
            logging.error(f"Drawing error: {str(e)}")
            self.running = False

    def draw_game_over(self):
        try:
            # Keep the last frame visible
            game_over_font = pygame.font.Font(None, 74)
            text = game_over_font.render('Game Over', True, self.RED)
            text_rect = text.get_rect(center=(self.game_width/2, self.game_height/2 - 50))
            
            score_text = self.font.render(f'Final Score: {self.score}', True, self.BLACK)
            score_rect = score_text.get_rect(center=(self.game_width/2, self.game_height/2 + 20))
            
            restart_text = self.font.render('Press R to Restart or ESC to Quit', True, self.GREEN)
            restart_rect = restart_text.get_rect(center=(self.game_width/2, self.game_height/2 + 70))

            self.game_screen.blit(text, text_rect)
            self.game_screen.blit(score_text, score_rect)
            self.game_screen.blit(restart_text, restart_rect)
            pygame.display.flip()

        except Exception as e:
            logging.error(f"Game over screen error: {str(e)}")

    def monitor_fps(self):
        self.fps_counter += 1
        current_time = pygame.time.get_ticks()
        if current_time - self.fps_timer > 1000:
            logging.debug(f"FPS: {self.fps_counter}")
            self.fps_counter = 0
            self.fps_timer = current_time

    def run(self):
        try:
            video_thread = threading.Thread(target=self.process_video)
            game_thread = threading.Thread(target=self.run_game)

            video_thread.start()
            game_thread.start()

            video_thread.join()
            game_thread.join()

        except Exception as e:
            logging.error(f"Threading error: {str(e)}")
        finally:
            self.cleanup()

    def cleanup(self):
        try:
            logging.info(f"Game ended. Final score: {self.score}")
            self.obstacles.clear()
            if self.cap is not None and self.cap.isOpened():
                self.cap.release()
            cv2.destroyAllWindows()
            pygame.quit()
        except Exception as e:
            logging.error(f"Cleanup error: {str(e)}")

if __name__ == "__main__":
    try:
        game = EndlessRunner()
        game.run()
    except Exception as e:
        logging.critical(f"Fatal error: {str(e)}")
        sys.exit(1)