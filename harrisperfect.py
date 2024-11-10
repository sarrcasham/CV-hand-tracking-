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
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7
        )

    def init_game_variables(self):
        self.player_width, self.player_height = 50, 50
        self.player_x = self.game_width // 4
        self.player_y = self.game_height // 2
        self.player_speed = 5
        self.obstacle_width, self.obstacle_height = 50, 50
        self.obstacle_speed = 5
        self.obstacles = []
        self.score = 0
        self.running = True
        self.game_state = "running"
        self.finger_y = self.game_height // 2
        self.last_finger_y = self.game_height // 2
        self.smoothing_factor = 0.7

    def detect_finger_position(self, frame):
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    index_finger_tip = hand_landmarks.landmark[
                        self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    y = int(index_finger_tip.y * self.game_height)
                    return self.smooth_movement(y)
            return self.finger_y
            
        except Exception as e:
            logging.error(f"Finger detection error: {str(e)}")
            return self.finger_y

    def smooth_movement(self, new_y):
        smoothed_y = (self.last_finger_y * self.smoothing_factor + 
                     new_y * (1 - self.smoothing_factor))
        self.last_finger_y = smoothed_y
        return int(smoothed_y)

    def cleanup(self):
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        pygame.quit()

class EndlessRunnerWithHarris(EndlessRunner):
    def __init__(self):
        super().__init__()
        # Harris corner parameters
        self.corner_threshold = 0.01
        self.blockSize = 2
        self.ksize = 3
        self.k = 0.04
        self.corner_points = []
        
        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)

    def detect_corners(self, frame):
        """Detect corners using Harris Corner Detection"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Convert to float32
            gray = np.float32(gray)
            
            # Apply Harris Corner Detection
            dst = cv2.cornerHarris(gray, self.blockSize, self.ksize, self.k)
            
            # Dilate to mark corners
            dst = cv2.dilate(dst, None)
            
            # Threshold for optimal points
            corner_frame = frame.copy()
            corner_points = []
            
            # Finding corners above threshold
            threshold = dst.max() * self.corner_threshold
            corner_coords = np.where(dst > threshold)
            
            for y, x in zip(*corner_coords):
                corner_points.append((x, y))
                cv2.circle(corner_frame, (x, y), 3, (0, 0, 255), -1)
                
            self.corner_points = corner_points
            return corner_frame
            
        except Exception as e:
            logging.error(f"Corner detection error: {str(e)}")
            return frame

    def run(self):
        """Main game loop"""
        try:
            # Create and start video processing thread
            video_thread = threading.Thread(target=self.process_video)
            video_thread.start()

            # Main game loop
            clock = pygame.time.Clock()
            while self.running:
                # Handle events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            self.running = False
                        elif event.key == pygame.K_r and self.game_state == "game_over":
                            self.init_game_variables()

                if self.game_state == "running":
                    # Update game state
                    self.update_game_state()
                    # Draw game
                    self.draw_game()
                elif self.game_state == "game_over":
                    self.draw_game_over()

                # Maintain frame rate
                clock.tick(60)

            # Clean up
            video_thread.join()
            self.cleanup()

        except Exception as e:
            logging.error(f"Game runtime error: {str(e)}")
            self.running = False

    def process_video(self):
        """Process video with both hand tracking and corner detection"""
        try:
            last_process_time = pygame.time.get_ticks()
            while self.running:
                current_time = pygame.time.get_ticks()
                if current_time - last_process_time < 16:  # ~60 FPS
                    continue
                    
                ret, frame = self.cap.read()
                if not ret:
                    continue

                frame = cv2.flip(frame, 1)
                
                # Perform hand detection
                self.finger_y = self.detect_finger_position(frame)
                
                # Perform Harris Corner detection
                corner_frame = self.detect_corners(frame)
                
                # Use corners for obstacle generation
                self.update_obstacles_from_corners()
                
                # Debug visualization
                cv2.imshow('Game View (with corners)', corner_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.running = False

                last_process_time = current_time
                
        except Exception as e:
            logging.error(f"Video processing error: {str(e)}")
            self.running = False

    def update_game_state(self):
        """Update game state including player and obstacles"""
        # Update player position
        self.player_y = min(max(self.finger_y, self.player_height), 
                          self.game_height - self.player_height)

        # Update obstacles
        for obstacle in self.obstacles[:]:  # Use slice to avoid modification during iteration
            obstacle.x -= self.obstacle_speed
            if obstacle.right < 0:
                self.obstacles.remove(obstacle)
                self.score += 1

        # Check collisions
        player_rect = pygame.Rect(self.player_x, self.player_y, 
                                self.player_width, self.player_height)
        for obstacle in self.obstacles:
            if player_rect.colliderect(obstacle):
                self.game_state = "game_over"
                return

    def draw_game(self):
        """Draw the game state"""
        # Fill background
        self.game_screen.fill(self.WHITE)

        # Draw player
        pygame.draw.rect(self.game_screen, self.BLACK,
                        (self.player_x, self.player_y, 
                         self.player_width, self.player_height))

        # Draw obstacles
        for obstacle in self.obstacles:
            pygame.draw.rect(self.game_screen, self.RED, obstacle)

        # Draw score
        font = pygame.font.Font(None, 36)
        score_text = font.render(f'Score: {self.score}', True, self.BLACK)
        self.game_screen.blit(score_text, (10, 10))

        # Update display
        pygame.display.flip()

    def draw_game_over(self):
        """Draw game over screen"""
        font = pygame.font.Font(None, 74)
        game_over_text = font.render('Game Over', True, self.RED)
        score_text = font.render(f'Score: {self.score}', True, self.BLACK)
        restart_text = font.render('Press R to Restart', True, self.GREEN)
        
        game_over_rect = game_over_text.get_rect(center=(self.game_width/2, self.game_height/2 - 50))
        score_rect = score_text.get_rect(center=(self.game_width/2, self.game_height/2 + 10))
        restart_rect = restart_text.get_rect(center=(self.game_width/2, self.game_height/2 + 70))
        
        self.game_screen.blit(game_over_text, game_over_rect)
        self.game_screen.blit(score_text, score_rect)
        self.game_screen.blit(restart_text, restart_rect)
        
        pygame.display.flip()

    def update_obstacles_from_corners(self):
        """Use detected corners to influence obstacle placement"""
        if not self.corner_points:
            return
            
        # Create clusters of corners using y-coordinates
        y_coords = [p[1] for p in self.corner_points]
        clusters = []
        current_cluster = [y_coords[0]]
        
        # Cluster corners based on vertical proximity
        threshold = 20  # pixels
        for y in sorted(y_coords[1:]):
            if abs(y - current_cluster[-1]) < threshold:
                current_cluster.append(y)
            else:
                clusters.append(current_cluster)
                current_cluster = [y]
                
        if current_cluster:
            clusters.append(current_cluster)
            
        # Use clusters for obstacle placement
        if random.random() < 0.02:  # Spawn rate
            if clusters:
                cluster = random.choice(clusters)
                avg_y = sum(cluster) / len(cluster)
                
                # Scale to game height
                scaled_y = (avg_y / self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) * self.game_height
                
                # Add randomness
                variation = random.randint(-30, 30)
                final_y = max(0, min(self.game_height - self.obstacle_height,
                                   scaled_y + variation))
                
                self.obstacles.append(
                    pygame.Rect(self.game_width, 
                              final_y,
                              self.obstacle_width, 
                              self.obstacle_height)
                )

if __name__ == "__main__":
    try:
        game = EndlessRunnerWithHarris()
        game.run()
    except Exception as e:
        logging.critical(f"Fatal error: {str(e)}")
        sys.exit(1)