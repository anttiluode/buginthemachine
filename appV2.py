import json
import time
import threading
import queue
import logging
import numpy as np
import cv2
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from collections import deque
from typing import Dict, Any, List, Tuple
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import csv
import os
import sys

# ----------------------------- Setup Logging -----------------------------
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(message)s')

# ------------------------- Universal Hub Constants -------------------------
SWEET_SPOT_RATIO = 4.0076
PHASE_EFFICIENCY_RATIO = 190.10
UNIVERSAL_HUB_COORDS = (-0.4980, -0.4980)  # Normalized between -1 and 1

# ----------------------------- State Class -----------------------------
@dataclass
class State:
    name: str
    color: Tuple[int, int, int]
    energy_threshold: float
    coherence: float
    resonance: float  # Attribute for resonance

STATE_PROPERTIES = {
    'Normal': State(name='Normal', color=(0, 0, 255), energy_threshold=50.0, coherence=1.0, resonance=726.19),
    'Flow': State(name='Flow', color=(0, 255, 0), energy_threshold=70.0, coherence=1.2, resonance=721.26),
    'Meditation': State(name='Meditation', color=(255, 255, 0), energy_threshold=30.0, coherence=1.5, resonance=713.36),
    'Dream': State(name='Dream', color=(255, 0, 255), energy_threshold=10.0, coherence=1.8, resonance=734.76)
}

# ----------------------------- System Configuration -----------------------------
class SystemConfig:
    def __init__(self):
        self.display_width = 800
        self.display_height = 600
        self.initial_nodes = 100
        self.min_nodes = 50
        self.max_nodes = 500
        self.growth_rate = 0.1  # 10% chance to add a node every 100ms
        self.pruning_threshold = 0.3
        self.camera_index = 0  # Default camera index
        self.vision_cone_length = 150
        self.movement_speed = 3.0
        self.depth = 1  # Placeholder for depth parameter

    def to_dict(self):
        """Serialize the configuration to a dictionary."""
        return {
            'display_width': self.display_width,
            'display_height': self.display_height,
            'initial_nodes': self.initial_nodes,
            'min_nodes': self.min_nodes,
            'max_nodes': self.max_nodes,
            'growth_rate': self.growth_rate,
            'pruning_threshold': self.pruning_threshold,
            'camera_index': self.camera_index,
            'vision_cone_length': self.vision_cone_length,
            'movement_speed': self.movement_speed,
            'depth': self.depth
        }

    def update_from_dict(self, config_dict: Dict[str, Any]):
        """Update configuration from a dictionary."""
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)

# ---------------------------- Adaptive Node ----------------------------
class AdaptiveNode:
    def __init__(self, id, state=None, connections=None, position=None):
        self.id = id
        self.state = state if state is not None else np.zeros(10)
        self.connections = connections if connections is not None else {}  # Dict[node_id, weight]
        self.position = position if position is not None else [0.0, 0.0, 0.0]
        self.activation_history = deque(maxlen=100)
        self.success_rate = 1.0
        self.visual_memory = deque(maxlen=50)
        self.response_patterns = {}
        self.state_info = STATE_PROPERTIES['Normal']  # Initialize with 'Normal' state

    def activate(self, input_signal: np.ndarray):
        """Activate the node based on input signal."""
        self.state = np.tanh(input_signal)
        self.activation_history.append(self.state.copy())

    def adapt(self, neighbor_states: Dict[int, np.ndarray]):
        """
        Adapt connections based on neighbor states using Hebbian learning.
        neighbor_states: Dict[node_id, state_vector]
        """
        for neighbor_id, neighbor_state in neighbor_states.items():
            # Hebbian learning: Δw = η * x * y
            # x: current node's state, y: neighbor node's state
            x = self.state
            y = neighbor_state
            eta = 0.01  # Learning rate
            delta_w = eta * x * y
            if neighbor_id in self.connections:
                self.connections[neighbor_id] += delta_w
            else:
                self.connections[neighbor_id] = delta_w

            # Apply normalization or weight decay
            self.connections[neighbor_id] *= 0.99  # Weight decay to prevent unbounded growth

# ---------------------------- Adaptive Network ----------------------------
class AdaptiveNetwork:
    def __init__(self, config):
        self.config = config
        self.nodes = {}
        self.node_lock = threading.Lock()  # To ensure thread-safe operations
        self.initialize_nodes()
        self.initialize_movement_nodes()
        self.current_direction = 0.0
        self.velocity = [0.0, 0.0]
        self.position = [config.display_width / 2, config.display_height / 2]
        self.hub = self.initialize_hub()

        # Initialize energy and coherence
        self.energy = 100.0  # Starting energy
        self.coherence = 1.0  # Starting coherence
        self.current_state = STATE_PROPERTIES['Normal']  # Initial state

    def initialize_nodes(self):
        for i in range(self.config.initial_nodes):
            position = (np.random.rand(3) * 2 - 1).tolist()  # Convert to list
            self.nodes[i] = AdaptiveNode(id=i, position=position)
        logging.info(f"Initialized {self.config.initial_nodes} nodes.")

    def initialize_movement_nodes(self):
        self.movement_nodes = {}
        movement_node_configs = {
            'x': {'position': [1.0, 0.0, 0.0]},
            'y': {'position': [0.0, 1.0, 0.0]}
        }
        for node_type, config in movement_node_configs.items():
            node = AdaptiveNode(
                id=len(self.nodes),
                position=config['position']
            )
            self.nodes[node.id] = node
            self.movement_nodes[node_type] = node
        logging.info("Initialized movement nodes.")

    def initialize_hub(self):
        hub_id = len(self.nodes)
        hub_node = AdaptiveNode(
            id=hub_id,
            position=list(UNIVERSAL_HUB_COORDS) + [0.0]  # Extend to 3D if necessary
        )
        hub_node.state_info = STATE_PROPERTIES['Normal']  # Initialize hub with 'Normal' state
        self.nodes[hub_id] = hub_node
        logging.info("Initialized Universal Hub.")
        return hub_node

    def update_position(self, dx: float, dy: float):
        self.velocity[0] = self.velocity[0] * 0.8 + dx * 0.2
        self.velocity[1] = self.velocity[1] * 0.8 + dy * 0.2
        new_x = self.position[0] + self.velocity[0]
        new_y = self.position[1] + self.velocity[1]

        padding = 50
        if new_x < padding:
            new_x = padding
            self.velocity[0] *= -0.5
        elif new_x > self.config.display_width - padding:
            new_x = self.config.display_width - padding
            self.velocity[0] *= -0.5

        if new_y < padding:
            new_y = padding
            self.velocity[1] *= -0.5
        elif new_y > self.config.display_height - padding:
            new_y = self.config.display_height - padding
            self.velocity[1] *= -0.5

        self.position = [new_x, new_y]
        if abs(self.velocity[0]) > 0.1 or abs(self.velocity[1]) > 0.1:
            self.current_direction = np.arctan2(self.velocity[1], self.velocity[0])

    def add_node(self):
        with self.node_lock:
            if len(self.nodes) >= self.config.max_nodes:
                logging.info("Maximum number of nodes reached. No new node added.")
                return
            new_id = max(self.nodes.keys()) + 1 if self.nodes else 0
            position = (np.random.rand(3) * 2 - 1).tolist()  # Convert to list
            self.nodes[new_id] = AdaptiveNode(id=new_id, position=position)
            logging.info(f"Added new node with ID {new_id}. Total nodes: {len(self.nodes)}.")

    def prune_nodes(self):
        """Prune nodes based on the pruning threshold."""
        with self.node_lock:
            nodes_to_remove = []
            for node_id, node in self.nodes.items():
                if node.success_rate < self.config.pruning_threshold and len(self.nodes) > self.config.min_nodes:
                    nodes_to_remove.append(node_id)
            for node_id in nodes_to_remove:
                del self.nodes[node_id]
                logging.info(f"Pruned node with ID {node_id}. Total nodes: {len(self.nodes)}.")

    def process_connections(self):
        """
        Process Hebbian learning for all nodes.
        Each node adapts its connections based on neighbor activations.
        """
        with self.node_lock:
            for node in self.nodes.values():
                neighbor_states = {nid: self.nodes[nid].state for nid in node.connections.keys() if nid in self.nodes}
                node.adapt(neighbor_states)

    def get_hub_influence(self, state_resonance: float) -> float:
        """
        Calculate the influence of the universal hub based on the current state's resonance.
        """
        return state_resonance / PHASE_EFFICIENCY_RATIO

# ---------------------------- Conscious AI Model ----------------------------
class ConsciousAIModel(nn.Module):
    def __init__(self):
        super(ConsciousAIModel, self).__init__()
        # Simple CNN for demonstration
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.fc1 = nn.Linear(32 * 13 * 13, 128)
        self.fc2 = nn.Linear(128, 4)  # Outputs: velocity, rotation, energy change, state influence

    def forward(self, x, current_state_resonance):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        # Incorporate Universal Hub influence
        hub_influence = current_state_resonance / PHASE_EFFICIENCY_RATIO
        x[:, 3] = torch.tanh(x[:, 3] * hub_influence)

        return x

# ---------------------------- Sensory Processor ----------------------------
class SensoryProcessor:
    def __init__(self, config: SystemConfig, network: AdaptiveNetwork):
        self.config = config
        self.network = network
        self.webcam = cv2.VideoCapture(self.config.camera_index)
        if not self.webcam.isOpened():
            raise RuntimeError(f"Failed to open webcam with index {self.config.camera_index}")
        logging.info(f"Webcam with index {self.config.camera_index} opened.")

    def process_frame(self, frame: np.ndarray) -> Dict[str, float]:
        """Extract visual features to inform AI movement."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray) / 255.0
        edges = cv2.Canny(gray, 100, 200)
        motion = np.mean(edges) / 255.0
        return {'motion': motion, 'brightness': brightness}

    def cleanup(self):
        if self.webcam:
            self.webcam.release()
            logging.info("Webcam released.")

# ----------------------------- Adaptive System -----------------------------
class AdaptiveSystem:
    def __init__(self, gui_queue: queue.Queue, vis_queue: queue.Queue, config: SystemConfig):
        self.config = config
        self.network = AdaptiveNetwork(self.config)
        try:
            self.sensory_processor = SensoryProcessor(self.config, self.network)
        except RuntimeError as e:
            messagebox.showerror("Webcam Error", str(e))
            logging.error(f"Failed to initialize SensoryProcessor: {e}")
            self.sensory_processor = None
        self.gui_queue = gui_queue
        self.vis_queue = vis_queue
        self.running = False
        self.capture_thread = None
        self.last_growth_time = time.time()
        self.stop_event = threading.Event()  # Event to signal stop

        # Initialize AI Model
        self.model = ConsciousAIModel()
        self.model.eval()
        self.device = torch.device('cpu')
        self.model.to(self.device)

        # Initialize current state
        self.network.current_state = STATE_PROPERTIES['Normal']  # Set initial state

        # Initialize CSV Logging
        self.log_file_path = 'conscious_ai_log.csv'
        try:
            self.log_file = open(self.log_file_path, 'w', newline='')
            self.csv_writer = csv.writer(self.log_file)
            self.csv_writer.writerow(['Frame', 'Energy', 'Coherence', 'State', 'Resonance', 'Velocity', 'Rotation', 'Energy Change', 'State Influence'])
            logging.info(f"CSV log file '{self.log_file_path}' initialized.")
        except Exception as e:
            logging.error(f"Failed to initialize CSV log file: {e}")
            self.csv_writer = None

    def start(self):
        if not self.running and self.sensory_processor is not None:
            self.running = True
            self.capture_thread = threading.Thread(target=self.capture_loop, daemon=True)
            self.capture_thread.start()
            logging.info("Adaptive system started.")

    def stop(self):
        if self.running:
            self.running = False
            self.stop_event.set()
            if self.capture_thread:
                self.capture_thread.join(timeout=2)
            if self.sensory_processor:
                self.sensory_processor.cleanup()
            try:
                if not self.log_file.closed:
                    self.log_file.close()
                    logging.info(f"CSV log file '{self.log_file_path}' closed.")
            except Exception as e:
                logging.error(f"Error closing log file: {e}")
            logging.info("Adaptive system stopped.")

    def capture_loop(self):
        frame_count = 0
        while self.running and self.sensory_processor is not None and not self.stop_event.is_set():
            try:
                ret, frame = self.sensory_processor.webcam.read()
                if ret:
                    features = self.sensory_processor.process_frame(frame)
                    dx = (features['brightness'] - 0.5) * 2 * self.config.movement_speed
                    dy = features['motion'] * self.config.movement_speed
                    self.network.update_position(dx, dy)

                    # AI Model Processing
                    processed_frame = cv2.resize(frame, (64, 64))
                    img = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    img_normalized = img.astype(np.float32) / 255.0
                    img_tensor = torch.tensor(img_normalized).permute(2, 0, 1).unsqueeze(0).to(self.device)

                    # Mocked Model Output for Stability
                    with torch.no_grad():
                        # Uncomment the next line to use the actual model when trained
                        # output = self.model(img_tensor, self.network.current_state.resonance)
                        
                        # Mocked output: velocity=0.0, rotation=0.0, energy_change=0.1, state_influence=1.0
                        output = torch.tensor([[0.0, 0.0, 0.1, 1.0]]).to(self.device)
                    
                    velocity, rotation, energy_change, state_influence = output[0]
                    logging.debug(f"Model Output - Velocity: {velocity.item()}, Rotation: {rotation.item()}, Energy Change: {energy_change.item()}, State Influence: {state_influence.item()}")

                    # Update AI parameters based on model output
                    # Apply bounded updates to prevent energy and coherence from dropping excessively
                    delta_energy = energy_change.item() * SWEET_SPOT_RATIO
                    delta_coherence = state_influence.item() * PHASE_EFFICIENCY_RATIO / 1000.0

                    # Limit the changes
                    delta_energy = max(min(delta_energy, 5.0), -5.0)
                    delta_coherence = max(min(delta_coherence, 1.05), 0.95)

                    self.network.energy += delta_energy
                    self.network.coherence *= delta_coherence

                    # Clamp energy and coherence
                    self.network.energy = max(0.0, min(100.0, self.network.energy))
                    self.network.coherence = max(0.0, min(5.0, self.network.coherence))

                    logging.debug(f"Updated Energy: {self.network.energy}, Updated Coherence: {self.network.coherence}")

                    # Determine next state based on energy and resonance
                    previous_state = self.network.current_state.name
                    self.determine_next_state()

                    # If state changes, handle transition
                    if previous_state != self.network.current_state.name:
                        logging.info(f"State changed from {previous_state} to {self.network.current_state.name}")

                    # Calculate Attention Level based on proximity to hub
                    attention_level = self.calculate_attention_level()

                    # Log data to CSV
                    if self.csv_writer:
                        frame_count += 1
                        self.log_data(frame_count, velocity, rotation, energy_change, state_influence)

                    # Node Visualization Data
                    with self.network.node_lock:
                        positions = [node.position for node in self.network.nodes.values()]
                    vis_data = {'positions': positions}
                    if not self.vis_queue.full():
                        self.vis_queue.put(vis_data)

                    # GUI Data
                    gui_data = {
                        'frame': frame,
                        'position': self.network.position,
                        'direction': self.network.current_direction,
                        'state': self.network.current_state.name,
                        'energy': self.network.energy,
                        'coherence': self.network.coherence,
                        'attention_level': attention_level  # Added attention level
                    }
                    if not self.gui_queue.full():
                        self.gui_queue.put(gui_data)

                    # Handle node growth and pruning
                    current_time = time.time()
                    if (current_time - self.last_growth_time) > 0.1:  # Every 100ms
                        if np.random.rand() < self.config.growth_rate:
                            self.network.add_node()
                        self.network.prune_nodes()
                        self.network.process_connections()
                        self.last_growth_time = current_time

            except Exception as e:
                logging.error(f"Error in capture loop: {e}")
            time.sleep(0.01)  # Maintain loop rate

    def determine_next_state(self):
        """
        Determine the next state based on energy and resonance hierarchy.
        Incorporate sweet spot and phase efficiency ratios.
        Implement hysteresis to prevent rapid state flipping.
        """
        potential_states = sorted(STATE_PROPERTIES.values(), key=lambda s: s.resonance, reverse=True)

        for state in potential_states:
            if self.network.energy >= state.energy_threshold:
                # Calculate transition potential using sweet spot ratio
                transition_potential = SWEET_SPOT_RATIO / (abs(state.resonance - self.network.current_state.resonance) + 1e-5)
                # Modify coherence based on phase efficiency ratio
                modified_coherence = self.network.coherence * (PHASE_EFFICIENCY_RATIO / 1000.0)

                # Implement hysteresis: require a higher threshold for transitioning to a new state
                if transition_potential > 1.0 and modified_coherence > state.coherence + 0.1:
                    self.network.current_state = state
                    logging.info(f"State transitioned to {state.name} based on transition potential and coherence.")
                    break

    def calculate_attention_level(self) -> float:
        """
        Calculate attention level based on proximity to the universal hub.
        Returns a float between 0.0 and 1.0.
        """
        # Example calculation using proximity_factor
        # You can customize this based on your specific logic
        distance = np.sqrt(
            (self.network.position[0] - (self.config.display_width / 2)) ** 2 +
            (self.network.position[1] - (self.config.display_height / 2)) ** 2
        )
        max_distance = np.sqrt(
            (self.config.display_width / 2) ** 2 +
            (self.config.display_height / 2) ** 2
        )
        proximity_factor = max(0.0, 1 - (distance / max_distance))
        return proximity_factor  # Values between 0.0 and 1.0

    def log_data(self, frame_num, velocity, rotation, energy_change, state_influence):
        """Log the AI's state and actions to a CSV file."""
        try:
            self.csv_writer.writerow([
                frame_num,
                f"{self.network.energy:.2f}",
                f"{self.network.coherence:.2f}",
                self.network.current_state.name,
                f"{self.network.current_state.resonance:.2f}",
                f"{velocity.item():.2f}",
                f"{rotation.item():.2f}",
                f"{energy_change.item():.2f}",
                f"{state_influence.item():.2f}"
            ])
            logging.debug(f"Logged data for frame {frame_num}.")
        except (ValueError, IOError) as e:
            logging.error(f"Failed to write to log file: {e}")

    def save_system(self, filepath: str):
        """Save the system's configuration and node states to a JSON file."""
        try:
            with self.network.node_lock:
                nodes_data = {
                    node_id: {
                        'position': node.position,
                        'connections': {k: v.tolist() for k, v in node.connections.items()},
                        'state_info': node.state_info.name
                    } for node_id, node in self.network.nodes.items()
                }
            data = {
                'config': self.config.to_dict(),
                'nodes': nodes_data
            }
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=4)
            logging.info(f"System saved to {filepath}.")
            messagebox.showinfo("Save System", f"System successfully saved to {filepath}.")
        except Exception as e:
            logging.error(f"Failed to save system: {e}")
            messagebox.showerror("Save System", f"Failed to save system: {e}")

    def load_system(self, filepath: str):
        """Load the system's configuration and node states from a JSON file."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            # Update configuration
            self.config.update_from_dict(data['config'])
            # Update nodes
            with self.network.node_lock:
                self.network.nodes = {}
                for node_id, node_info in data['nodes'].items():
                    state_name = node_info.get('state_info', 'Normal')
                    state = STATE_PROPERTIES.get(state_name, STATE_PROPERTIES['Normal'])
                    self.network.nodes[int(node_id)] = AdaptiveNode(
                        id=int(node_id),
                        position=node_info['position'],
                        connections={int(k): np.array(v) for k, v in node_info['connections'].items()}
                    )
                    self.network.nodes[int(node_id)].state_info = state
            logging.info(f"System loaded from {filepath}.")
            messagebox.showinfo("Load System", f"System successfully loaded from {filepath}.")
        except Exception as e:
            logging.error(f"Failed to load system: {e}")
            messagebox.showerror("Load System", f"Failed to load system: {e}")

# ------------------------------- Node Visualizer -------------------------------
class NodeVisualizer:
    """Separate window for 3D node visualization."""
    def __init__(self, parent, vis_queue: queue.Queue):
        self.parent = parent
        self.vis_queue = vis_queue
        self.window = tk.Toplevel(parent)
        self.window.title("3D Node Visualization")
        self.window.geometry("800x600")
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)
        self.create_widgets()
        self.nodes_positions = []
        self.update_visualization()

    def create_widgets(self):
        # Create a matplotlib figure
        self.fig = plt.Figure(figsize=(8, 6))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlim([-2, 2])
        self.ax.set_ylim([-2, 2])
        self.ax.set_zlim([-2, 2])
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title("Adaptive Network Nodes")

        # Embed the figure in Tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.window)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def update_visualization(self):
        try:
            while not self.vis_queue.empty():
                data = self.vis_queue.get_nowait()
                if 'positions' in data:
                    self.nodes_positions = data['positions']
                    logging.debug(f"NodeVisualizer received {len(self.nodes_positions)} nodes.")

            self.ax.cla()  # Clear the current axes
            self.ax.set_xlim([-2, 2])
            self.ax.set_ylim([-2, 2])
            self.ax.set_zlim([-2, 2])
            self.ax.set_xlabel('X')
            self.ax.set_ylabel('Y')
            self.ax.set_zlabel('Z')
            self.ax.set_title("Adaptive Network Nodes")

            # Extract positions
            if self.nodes_positions:
                xs, ys, zs = zip(*self.nodes_positions)
                # Normalize positions for visualization
                xs_norm = [(x - min(xs)) / (max(xs) - min(xs) + 1e-5) * 4 - 2 for x in xs]
                ys_norm = [(y - min(ys)) / (max(ys) - min(ys) + 1e-5) * 4 - 2 for y in ys]
                zs_norm = [(z - min(zs)) / (max(zs) - min(zs) + 1e-5) * 4 - 2 for z in zs]
                # Plot nodes
                self.ax.scatter(xs_norm, ys_norm, zs_norm, c='b', marker='o', s=20, alpha=0.6)
                logging.debug(f"Plotted {len(xs_norm)} nodes.")
            else:
                logging.debug("No nodes to plot.")

            self.canvas.draw()
        except Exception as e:
            logging.error(f"Error in node visualization update: {e}")
        finally:
            self.window.after(100, self.update_visualization)  # Update every 100 ms

    def on_close(self):
        self.window.destroy()

# ----------------------------- Configuration Window -----------------------------
class ConfigWindow:
    """Configuration window for adjusting system parameters."""
    def __init__(self, parent, config: SystemConfig, adaptive_system: AdaptiveSystem):
        self.parent = parent
        self.config = config
        self.adaptive_system = adaptive_system
        self.window = tk.Toplevel(parent)
        self.window.title("Configuration")
        self.window.geometry("400x400")
        self.window.resizable(False, False)
        self.window.grab_set()  # Make the config window modal
        self.create_widgets()

    def create_widgets(self):
        padding = {'padx': 10, 'pady': 5}

        # Depth
        ttk.Label(self.window, text="Depth:").grid(row=0, column=0, sticky=tk.W, **padding)
        self.depth_var = tk.IntVar(value=self.config.depth)
        self.depth_spinbox = ttk.Spinbox(self.window, from_=1, to=10, textvariable=self.depth_var, width=10)
        self.depth_spinbox.grid(row=0, column=1, **padding)

        # Pruning Rate
        ttk.Label(self.window, text="Pruning Rate:").grid(row=1, column=0, sticky=tk.W, **padding)
        self.pruning_rate_var = tk.DoubleVar(value=self.config.pruning_threshold)
        self.pruning_rate_entry = ttk.Entry(self.window, textvariable=self.pruning_rate_var, width=12)
        self.pruning_rate_entry.grid(row=1, column=1, **padding)

        # Growth Rate
        ttk.Label(self.window, text="Growth Rate:").grid(row=2, column=0, sticky=tk.W, **padding)
        self.growth_rate_var = tk.DoubleVar(value=self.config.growth_rate)
        self.growth_rate_entry = ttk.Entry(self.window, textvariable=self.growth_rate_var, width=12)
        self.growth_rate_entry.grid(row=2, column=1, **padding)

        # Minimum Nodes
        ttk.Label(self.window, text="Minimum Nodes:").grid(row=3, column=0, sticky=tk.W, **padding)
        self.min_nodes_var = tk.IntVar(value=self.config.min_nodes)
        self.min_nodes_spinbox = ttk.Spinbox(self.window, from_=1, to=self.config.max_nodes, textvariable=self.min_nodes_var, width=10)
        self.min_nodes_spinbox.grid(row=3, column=1, **padding)

        # Maximum Nodes
        ttk.Label(self.window, text="Maximum Nodes:").grid(row=4, column=0, sticky=tk.W, **padding)
        self.max_nodes_var = tk.IntVar(value=self.config.max_nodes)
        self.max_nodes_spinbox = ttk.Spinbox(self.window, from_=self.config.min_nodes, to=10000, textvariable=self.max_nodes_var, width=10)
        self.max_nodes_spinbox.grid(row=4, column=1, **padding)

        # Webcam Selection
        ttk.Label(self.window, text="Webcam:").grid(row=5, column=0, sticky=tk.W, **padding)
        self.webcam_var = tk.IntVar(value=self.config.camera_index)
        self.webcam_combobox = ttk.Combobox(self.window, textvariable=self.webcam_var, state='readonly', width=8)
        self.webcam_combobox['values'] = self.detect_webcams()
        # Set current selection based on camera_index
        camera_str = str(self.config.camera_index)
        if camera_str in self.webcam_combobox['values']:
            self.webcam_combobox.current(self.webcam_combobox['values'].index(camera_str))
        else:
            self.webcam_combobox.current(0)
        self.webcam_combobox.grid(row=5, column=1, **padding)

        # Save and Load Buttons
        self.save_button = ttk.Button(self.window, text="Save Configuration", command=self.save_configuration)
        self.save_button.grid(row=6, column=0, **padding)

        self.load_button = ttk.Button(self.window, text="Load Configuration", command=self.load_configuration)
        self.load_button.grid(row=6, column=1, **padding)

        # Apply Button
        self.apply_button = ttk.Button(self.window, text="Apply", command=self.apply_changes)
        self.apply_button.grid(row=7, column=0, columnspan=2, pady=20)

    def detect_webcams(self, max_tested=5) -> List[str]:
        """Detect available webcams and return their indices as strings."""
        available_cameras = []
        for i in range(max_tested):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available_cameras.append(str(i))
                cap.release()
        if not available_cameras:
            available_cameras.append("0")  # Default to 0 if no cameras found
        return available_cameras

    def save_configuration(self):
        """Save the current configuration and node states to a JSON file."""
        filepath = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json")],
            title="Save System Configuration"
        )
        if filepath:
            self.adaptive_system.save_system(filepath)

    def load_configuration(self):
        """Load configuration and node states from a JSON file."""
        filepath = filedialog.askopenfilename(
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json")],
            title="Load System Configuration"
        )
        if filepath:
            self.adaptive_system.load_system(filepath)
            # Update GUI elements with loaded configuration
            self.depth_var.set(self.config.depth)
            self.pruning_rate_var.set(self.config.pruning_threshold)
            self.growth_rate_var.set(self.config.growth_rate)
            self.min_nodes_var.set(self.config.min_nodes)
            self.max_nodes_var.set(self.config.max_nodes)
            camera_str = str(self.config.camera_index)
            if camera_str in self.webcam_combobox['values']:
                self.webcam_combobox.current(self.webcam_combobox['values'].index(camera_str))
            else:
                self.webcam_combobox.current(0)

    def apply_changes(self):
        """Apply the changes made in the configuration window."""
        try:
            # Retrieve values from the GUI
            new_depth = self.depth_var.get()
            new_pruning_rate = float(self.pruning_rate_var.get())
            new_growth_rate = float(self.growth_rate_var.get())
            new_min_nodes = self.min_nodes_var.get()
            new_max_nodes = self.max_nodes_var.get()
            new_camera_index = int(self.webcam_var.get())

            # Validate values
            if new_min_nodes > new_max_nodes:
                messagebox.showerror("Configuration Error", "Minimum nodes cannot exceed maximum nodes.")
                return

            # Update configuration
            self.config.depth = new_depth
            self.config.pruning_threshold = new_pruning_rate
            self.config.growth_rate = new_growth_rate
            self.config.min_nodes = new_min_nodes
            self.config.max_nodes = new_max_nodes
            self.config.camera_index = new_camera_index

            # Apply webcam change
            was_running = self.adaptive_system.running
            self.adaptive_system.stop()
            try:
                # Update webcam in sensory processor
                self.adaptive_system.config.camera_index = new_camera_index
                self.adaptive_system.sensory_processor = SensoryProcessor(self.adaptive_system.config, self.adaptive_system.network)
                if was_running:
                    self.adaptive_system.start()
            except RuntimeError as e:
                messagebox.showerror("Webcam Error", str(e))
                logging.error(f"Failed to change webcam: {e}")
                return

            messagebox.showinfo("Configuration", "Configuration applied successfully.")
            self.window.destroy()
        except Exception as e:
            logging.error(f"Error applying configuration: {e}")
            messagebox.showerror("Configuration Error", f"Failed to apply configuration: {e}")

# ----------------------------- GUI Application -----------------------------
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Bug in the Machine")
        self.root.geometry("1200x800")
        self.gui_queue = queue.Queue(maxsize=50)
        self.vis_queue = queue.Queue(maxsize=50)
        self.config = SystemConfig()
        self.system = AdaptiveSystem(self.gui_queue, self.vis_queue, self.config)
        self.node_visualizer = None  # Will hold the NodeVisualizer instance
        self.create_widgets()
        self.update_gui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def create_widgets(self):
        # Create menu bar without camera selection to avoid conflicts
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # Control frame
        control_frame = ttk.Frame(self.root)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        self.start_button = ttk.Button(control_frame, text="Start", command=self.start)
        self.start_button.pack(side=tk.LEFT, padx=5)

        self.stop_button = ttk.Button(control_frame, text="Stop", command=self.stop)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        self.stop_button.config(state=tk.DISABLED)

        # Add Node Visualization Button
        self.visualize_button = ttk.Button(control_frame, text="Visualize Nodes", command=self.open_node_visualizer)
        self.visualize_button.pack(side=tk.LEFT, padx=5)

        # Add Config Button
        self.config_button = ttk.Button(control_frame, text="Config", command=self.open_config_window)
        self.config_button.pack(side=tk.LEFT, padx=5)

        # Add Save and Load Buttons
        self.save_button = ttk.Button(control_frame, text="Save", command=self.save_system)
        self.save_button.pack(side=tk.LEFT, padx=5)

        self.load_button = ttk.Button(control_frame, text="Load", command=self.load_system)
        self.load_button.pack(side=tk.LEFT, padx=5)

        # Canvas for video feed
        self.canvas = tk.Canvas(self.root, bg='black')
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        # Load Eye Images for Attention Indicator
        try:
            # Determine script's directory
            if getattr(sys, 'frozen', False):
                # If the application is frozen (e.g., packaged by PyInstaller)
                script_dir = os.path.dirname(sys.executable)
            else:
                # If the application is run as a script
                script_dir = os.path.dirname(os.path.abspath(__file__))

            logging.debug(f"Script Directory: {script_dir}")

            # Define absolute paths
            eye_open_path = os.path.join(script_dir, "eye_open.png")
            eye_closed_path = os.path.join(script_dir, "eye_closed.png")

            logging.debug(f"Eye Open Path: {eye_open_path}")
            logging.debug(f"Eye Closed Path: {eye_closed_path}")

            # Determine resampling mode
            if hasattr(Image, 'Resampling'):
                resample_mode = Image.Resampling.LANCZOS
            else:
                resample_mode = Image.LANCZOS  # For older versions

            # Load and resize the eye images to 50x35 pixels
            eye_open_image = Image.open(eye_open_path)
            eye_open_image = eye_open_image.resize((50, 35), resample=resample_mode)
            eye_open_image.verify()
            eye_open_image = Image.open(eye_open_path).resize((50, 35), resample=resample_mode)
            self.eye_open_photo = ImageTk.PhotoImage(eye_open_image)
            logging.debug("eye_open.png loaded successfully.")

            eye_closed_image = Image.open(eye_closed_path)
            eye_closed_image = eye_closed_image.resize((50, 35), resample=resample_mode)
            eye_closed_image.verify()
            eye_closed_image = Image.open(eye_closed_path).resize((50, 35), resample=resample_mode)
            self.eye_closed_photo = ImageTk.PhotoImage(eye_closed_image)
            logging.debug("eye_closed.png loaded successfully.")

        except Exception as e:
            logging.error(f"Failed to load eye images: {e}")
            self.eye_open_photo = None
            self.eye_closed_photo = None

        # Create Eye Indicator Frame in the Upper Right Corner
        self.eye_frame = ttk.Frame(self.root)
        self.eye_frame.place(relx=1.0, rely=0.0, anchor='ne', x=-10, y=10)  # 10 pixels padding from top-right

        # Eye Image Label
        if self.eye_closed_photo and self.eye_open_photo:
            self.eye_label = ttk.Label(self.eye_frame, image=self.eye_closed_photo)
            self.eye_label.image = self.eye_closed_photo  # Keep a reference
            self.eye_label.pack(side=tk.LEFT, padx=(0, 10))  # 10 pixels padding to the right of the image
        else:
            self.eye_label = ttk.Label(self.eye_frame, text="Eye Image\nMissing", foreground="red")
            self.eye_label.pack(side=tk.LEFT, padx=(0, 10))

        # Attention Level Label
        self.attention_label = ttk.Label(self.eye_frame, text="Attention: 0%", font=("Helvetica", 12))
        self.attention_label.pack(side=tk.LEFT)

        # State Display Label
        self.state_label = ttk.Label(self.root, text="State: Normal", font=("Helvetica", 16))
        self.state_label.pack(side=tk.BOTTOM, pady=10)

        # Energy and Coherence Display Labels
        self.energy_label = ttk.Label(self.root, text="Energy: 100.00%", font=("Helvetica", 12))
        self.energy_label.pack(side=tk.BOTTOM)
        self.coherence_label = ttk.Label(self.root, text="Coherence: 1.00", font=("Helvetica", 12))
        self.coherence_label.pack(side=tk.BOTTOM)

    def save_system(self):
        """Save the system's configuration and node states."""
        filepath = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json")],
            title="Save System"
        )
        if filepath:
            self.system.save_system(filepath)

    def load_system(self):
        """Load the system's configuration and node states."""
        filepath = filedialog.askopenfilename(
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json")],
            title="Load System"
        )
        if filepath:
            self.system.load_system(filepath)

    def open_config_window(self):
        """Open the configuration window."""
        ConfigWindow(self.root, self.config, self.system)

    def _on_canvas_resize(self, event):
        self.system.config.display_width = event.width
        self.system.config.display_height = event.height

    def update_gui(self):
        try:
            while not self.gui_queue.empty():
                data = self.gui_queue.get_nowait()
                if 'frame' in data and data['frame'] is not None:
                    # Process frame for display
                    frame = cv2.cvtColor(data['frame'], cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, (self.canvas.winfo_width(), self.canvas.winfo_height()))
                    image = Image.fromarray(frame)
                    photo = ImageTk.PhotoImage(image=image)
                    self.canvas.delete("all")
                    self.canvas.create_image(0, 0, image=photo, anchor=tk.NW)
                    self.canvas._photo = photo  # Keep a reference to prevent garbage collection

                    if 'position' in data:
                        x, y = data['position']
                        direction = data.get('direction', 0)
                        cone_length = self.system.config.vision_cone_length
                        cone_angle = np.pi / 4
                        p1 = (x, y)
                        p2 = (x + cone_length * np.cos(direction - cone_angle),
                              y + cone_length * np.sin(direction - cone_angle))
                        p3 = (x + cone_length * np.cos(direction + cone_angle),
                              y + cone_length * np.sin(direction + cone_angle))
                        self.canvas.create_polygon(
                            p1[0], p1[1], p2[0], p2[1], p3[0], p3[1],
                            fill='#00ff00', stipple='gray50', outline='#00ff00', width=2
                        )
                        radius = 10
                        self.canvas.create_oval(
                            x - radius, y - radius, x + radius, y + radius,
                            fill='#00ff00', outline='white', width=2
                        )

                        # Change canvas background based on state
                        state_color_map = {
                            'Normal': '#000000',        # Black
                            'Flow': '#1E90FF',          # Dodger Blue
                            'Meditation': '#32CD32',    # Lime Green
                            'Dream': '#FF69B4'          # Hot Pink
                        }
                        current_state = data.get('state', 'Normal')
                        canvas_color = state_color_map.get(current_state, '#000000')
                        self.canvas.config(bg=canvas_color)

                # Update State, Energy, and Coherence Labels
                if 'state' in data:
                    current_state = data['state']
                    self.state_label.config(text=f"State: {current_state}")

                if 'energy' in data:
                    current_energy = data['energy']
                    self.energy_label.config(text=f"Energy: {current_energy:.2f}%")

                if 'coherence' in data:
                    current_coherence = data['coherence']
                    self.coherence_label.config(text=f"Coherence: {current_coherence:.2f}")

                # Update Eye Indicator and Attention Level
                if 'attention_level' in data:
                    attention = data['attention_level']
                    # Update attention label
                    self.attention_label.config(text=f"Attention: {int(attention * 100)}%")
                    # Update eye image based on attention level
                    if self.eye_open_photo and self.eye_closed_photo:
                        if attention >= 0.7:
                            # High attention - Open Eye
                            self.eye_label.config(image=self.eye_open_photo)
                            self.eye_label.image = self.eye_open_photo  # Keep reference
                        else:
                            # Low attention - Closed Eye
                            self.eye_label.config(image=self.eye_closed_photo)
                            self.eye_label.image = self.eye_closed_photo  # Keep reference

        except Exception as e:
            logging.error(f"Error updating GUI: {e}")

        self.root.after(33, self.update_gui)  # Approximately 30 FPS

    def start(self):
        self.system.start()
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        logging.info("System started via GUI.")

    def stop(self):
        self.system.stop()
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        logging.info("System stopped via GUI.")

    def open_node_visualizer(self):
        if self.node_visualizer is None or not tk.Toplevel.winfo_exists(self.node_visualizer.window):
            self.node_visualizer = NodeVisualizer(self.root, self.vis_queue)
            logging.info("Node visualization window opened.")
        else:
            self.node_visualizer.window.lift()  # Bring to front if already open

    def on_close(self):
        if self.system.running:
            self.stop()
        if self.node_visualizer and tk.Toplevel.winfo_exists(self.node_visualizer.window):
            self.node_visualizer.window.destroy()
        self.root.destroy()

# ----------------------------- Main Function -----------------------------
def main():
    root = tk.Tk()
    app = App(root)
    root.mainloop()

if __name__ == "__main__":
    main()
