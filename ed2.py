import cv2
import numpy as np
import time
import logging
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import json
from config import LOG_FORMAT, FUSION_STRATEGY, DETECTION_INTERVAL, TEMPORAL_SMOOTHING_WINDOW, DETECTION_INTERVAL, TEMPORAL_SMOOTHING_WINDOW
from collections import deque
from model_loader import ModelLoader
from video_processor import VideoProcessor
from audio_processor import AudioProcessor
from emotion_fusion import EmotionFusion

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("emotion_gui")

class AdvancedEmotionDetectionGUI:
    def __init__(self, root, camera_index=0, fusion_strategy=FUSION_STRATEGY):
        self.root = root
        self.root.title("🎥 Real-Time Emotion Detection Dashboard")
        self.root.geometry("1000x700")
        self.root.configure(bg="#2e2e2e")

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TButton", font=("Arial", 11), padding=6)

        self.camera_index = camera_index
        self.is_running = False
        self.frame_count = 0
        self.start_time = None
        self.fps = 50
        self.current_emotion = "Neutral"
        self.current_confidence = 0.0
        
        # Temporal smoothing for accuracy improvement
        self.emotion_history = []
        self.confidence_history = []
        self.smoothing_window = TEMPORAL_SMOOTHING_WINDOW
        self._smooth_window = []  # For backward compatibility
        self._smooth_window = deque(maxlen=TEMPORAL_SMOOTHING_WINDOW)

        # Core systems
        self.model_loader = ModelLoader()
        self.fusion = EmotionFusion(fusion_strategy)

        # Notebook (tabs)
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True)

        # Video tab
        self.tab_video = ttk.Frame(notebook)
        notebook.add(self.tab_video, text="📷 Live Video")
        self.label_video = tk.Label(self.tab_video, bg="black")
        self.label_video.pack(fill="both", expand=True)

        # Stats tab
        self.tab_stats = ttk.Frame(notebook)
        notebook.add(self.tab_stats, text="📊 Emotion Stats")
        self.label_status = tk.Label(self.tab_stats, text="Status: Ready", bg="#2e2e2e", fg="white", font=("Arial", 13))
        self.label_status.pack(pady=10)
        self.progress = ttk.Progressbar(self.tab_stats, length=400, mode="determinate")
        self.progress.pack(pady=10)

        fig = Figure(figsize=(5, 3), dpi=100)
        self.ax = fig.add_subplot(111)
        self.ax.set_title("Emotion History")
        self.ax.set_ylim(0, 1)
        self.line, = self.ax.plot([], [], color="cyan")
        self.history_x, self.history_y = [], []
        self.canvas = FigureCanvasTkAgg(fig, master=self.tab_stats)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # Logs tab
        self.tab_logs = ttk.Frame(notebook)
        notebook.add(self.tab_logs, text="📝 Logs")
        self.text_logs = tk.Text(self.tab_logs, bg="black", fg="white", font=("Consolas", 10))
        self.text_logs.pack(fill="both", expand=True)

        # Buttons
        btns = ttk.Frame(self.root)
        btns.pack(pady=5)
        ttk.Button(btns, text="▶ Start", command=self.run).grid(row=0, column=0, padx=5)
        ttk.Button(btns, text="⏸ Stop", command=self.stop).grid(row=0, column=1, padx=5)
        ttk.Button(btns, text="💾 Save Stats", command=self._save_statistics).grid(row=0, column=2, padx=5)
        ttk.Button(btns, text="♻ Reset Stats", command=self._reset_statistics).grid(row=0, column=3, padx=5)
        ttk.Button(btns, text="❌ Quit", command=self.cleanup).grid(row=0, column=4, padx=5)

        # Load models AFTER UI is ready, then create processors with loaded models
        self._load_models()
        self.video_processor = VideoProcessor(model=self.model_loader.video_model, camera_index=camera_index)
        self.audio_processor = AudioProcessor(model=self.model_loader.audio_model)

    def log_message(self, msg):
        print(msg)  # Always print to console for debugging
        if hasattr(self, "text_logs"):  # Only log if widget exists
            self.text_logs.insert(tk.END, msg + "\n")
            self.text_logs.see(tk.END)

    def _load_models(self):
        v, a = self.model_loader.load_all_models()
        if not v and not a:
            self.log_message("⚠ No models found, using dummy models")
            self.model_loader.create_dummy_models()
        self.log_message(f"Model status: {self.model_loader.get_model_info()}")

    def _calculate_fps(self):
        self.frame_count += 1
        if self.start_time is None:
            self.start_time = time.time()
        if self.frame_count % 30 == 0:
            elapsed = time.time() - self.start_time
            self.fps = self.frame_count / elapsed

    def _process_frame(self, frame):
        # Use detection interval to skip expensive face detection on some frames
        # This significantly increases FPS while maintaining accuracy with tracking
        detect_faces_now = (self.frame_count % max(1, DETECTION_INTERVAL) == 0)
        
        # Get faces using tracking (faster) or full detection (more accurate but slower)
        faces = self.video_processor.get_tracked_faces(frame, detect=detect_faces_now)
        
        emotions = []
        for (x, y, w, h) in faces:
            roi = frame[y:y+h, x:x+w]
            emo, conf = self.video_processor.get_face_emotion(roi)
            emotions.append((emo, conf))
        
        audio_emo, audio_conf = self.audio_processor.get_audio_emotion()
        
        if emotions:
            video_emo, video_conf = emotions[0]
            fused_emo, fused_conf = self.fusion.fuse_emotions(video_emo, video_conf, audio_emo, audio_conf)
        else:
            fused_emo, fused_conf = audio_emo, audio_conf
        
        # Apply temporal smoothing for better accuracy and stability
        self.emotion_history.append(fused_emo)
        self.confidence_history.append(fused_conf)
        self._smooth_window.append(fused_conf)  # For backward compatibility
        
        # Keep only recent history for smoothing
        if len(self.emotion_history) > self.smoothing_window:
            self.emotion_history.pop(0)
            self.confidence_history.pop(0)
        if len(self._smooth_window) > self.smoothing_window:
            self._smooth_window.pop(0)
        
        # Use majority vote for emotion and average for confidence (temporal smoothing)
        if len(self.emotion_history) >= 3:
            # Get most common emotion in recent history
            from collections import Counter
            emotion_counts = Counter(self.emotion_history[-self.smoothing_window:])
            smoothed_emo = emotion_counts.most_common(1)[0][0]
            
            # Average confidence over recent history
            smoothed_conf = sum(self.confidence_history[-self.smoothing_window:]) / len(self.confidence_history[-self.smoothing_window:])
            
            self.current_emotion = smoothed_emo
            self.current_confidence = smoothed_conf
        else:
            self.current_emotion = fused_emo
            self.current_confidence = fused_conf
        
        return self._draw_annotations(frame, faces, emotions, audio_emo, audio_conf, self.current_emotion, self.current_confidence)

    def _draw_annotations(self, frame, faces, emotions, audio_emo, audio_conf, fused_emo, fused_conf):
        for i, (x, y, w, h) in enumerate(faces):
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            if i < len(emotions):
                emo, conf = emotions[i]
                label_y = max(y - 10, 20)
                cv2.putText(frame, f"Face: {emo} ({conf:.2f})", (x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(frame, f"Audio: {audio_emo} ({audio_conf:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        cv2.putText(frame, f"Final: {fused_emo} ({fused_conf:.2f})", (10, 60), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0,255,0), 2)
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        return frame

    def _update_stats(self):
        self.label_status.config(text=f"FPS: {self.fps:.1f} | Emotion: {self.current_emotion} ({self.current_confidence:.2f})")
        self.progress["value"] = self.current_confidence * 100
        self.history_x.append(self.frame_count)
        self.history_y.append(self.current_confidence)
        self.line.set_data(self.history_x, self.history_y)
        self.ax.set_xlim(0, max(50, self.frame_count))
        self.canvas.draw_idle()

    def run(self):
        if not self.video_processor.cap:
            self.video_processor.start_capture()
        self.audio_processor.start_recording()
        self.is_running = True
        self.update_frame()
        self.log_message("▶ Detection started")

    def update_frame(self):
        if not self.is_running:
            return
        success, frame = self.video_processor.read_frame()
        if success:
            frame = self._process_frame(frame)
            self._calculate_fps()
            
            # Update stats less frequently to improve FPS (every 2 frames)
            if self.frame_count % 2 == 0:
                self._update_stats()
            
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.label_video.imgtk = img
            self.label_video.config(image=img)
        # Use 0ms delay for maximum frame rate
        self.root.after(0, self.update_frame)

    def stop(self):
        self.is_running = False
        self.log_message("⏸ Detection stopped")

    def _save_statistics(self):
        try:
            stats = self.fusion.get_emotion_statistics()
            with open("emotion_stats.json", "w") as f:
                json.dump(stats, f, indent=2)
            self.log_message("💾 Statistics saved to emotion_stats.json")
        except Exception as e:
            self.log_message(f"❌ Error saving statistics: {e}")

    def _reset_statistics(self):
        self.fusion.emotion_history.clear()
        self.frame_count, self.start_time, self.fps = 0, None, 0
        self.history_x.clear()
        self.history_y.clear()
        self.emotion_history.clear()
        self.confidence_history.clear()
        self._smooth_window.clear()
        self.progress["value"] = 0
        self.canvas.draw_idle()
        self.log_message("♻ Statistics reset")

    def cleanup(self):
        self.is_running = False
        self.video_processor.cleanup()
        self.audio_processor.cleanup()
        self.model_loader.cleanup()
        self.log_message("❌ Clean exit")
        self.root.destroy()


def main():
    root = tk.Tk()
    app = AdvancedEmotionDetectionGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.cleanup)
    root.mainloop()

if __name__ == "__main__":
    main()