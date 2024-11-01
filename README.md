# Bug in the Machine


## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
  - [System Configuration](#system-configuration)
  - [Adaptive Network](#adaptive-network)
  - [Sensory Processing](#sensory-processing)
  - [Visualization](#visualization)
  - [Graphical User Interface](#graphical-user-interface)
- [Installation](#installation)
- [Usage](#usage)
  - [Starting the Application](#starting-the-application)
  - [Configuration](#configuration)
  - [Visualization](#visualization)
  - [Saving and Loading System States](#saving-and-loading-system-states)
- [Dependencies](#dependencies)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Overview

**Bug in the Machine** is an interactive Python application that combines computer vision, adaptive neural networks, and real-time visualization to create a dynamic and responsive system. Leveraging a webcam for sensory input, the application processes visual data to influence the behavior and structure of an internal adaptive network. This network can dynamically grow or prune nodes based on predefined configurations and real-time inputs, providing a tangible demonstration of adaptive systems and machine learning principles.

The application offers a user-friendly graphical interface (GUI) built with Tkinter, allowing users to start or stop the system, adjust configurations, visualize the adaptive network in 3D, and manage system states through saving and loading functionalities.

---

## Features

- **Real-Time Video Processing:** Captures live video feed using a webcam and extracts key visual features such as brightness and motion.
- **Adaptive Neural Network:** Implements a network of nodes that can dynamically grow or prune based on system configurations and sensory inputs.
- **Hebbian Learning:** Utilizes Hebbian learning principles to adapt connections between nodes, enabling self-organizing behavior.
- **3D Visualization:** Offers a separate window to visualize the adaptive network in 3D space, updating in real-time as the network evolves.
- **Configuration Management:** Provides a configuration window for adjusting system parameters like node counts, growth rates, pruning thresholds, and webcam selection.
- **Data Persistence:** Allows saving and loading of system configurations and network states to JSON files, facilitating easy state management.
- **User-Friendly GUI:** Built with Tkinter, featuring intuitive controls for starting/stopping the system, accessing configurations, and managing visualizations.

---

## Architecture

**Bug in the Machine** is structured into several modular components, each responsible for distinct functionalities. This modularity ensures scalability, maintainability, and ease of understanding.

### System Configuration

- **Class:** `SystemConfig`
- **Purpose:** Centralizes all adjustable parameters of the system, facilitating easy serialization and deserialization of settings.
- **Key Attributes:**
  - Display settings (`display_width`, `display_height`)
  - Network parameters (`initial_nodes`, `min_nodes`, `max_nodes`, `growth_rate`, `pruning_threshold`)
  - Camera settings (`camera_index`)
  - Movement and vision parameters (`vision_cone_length`, `movement_speed`)
  - Depth parameter (`depth`)

### Adaptive Network

- **Classes:** `AdaptiveNode`, `AdaptiveNetwork`
- **Purpose:** Manages a collection of adaptive nodes that form the neural network. Nodes can dynamically grow or prune based on system configurations and sensory inputs.
- **Key Features:**
  - **Dynamic Growth:** Nodes are added probabilistically based on the `growth_rate`.
  - **Pruning:** Nodes with low performance (`success_rate`) are removed to maintain optimal network size.
  - **Hebbian Learning:** Connections between nodes are adjusted based on their activation states, promoting adaptive behavior.

### Sensory Processing

- **Class:** `SensoryProcessor`
- **Purpose:** Interfaces with the webcam to capture video frames and extract meaningful visual features that influence the adaptive network's behavior.
- **Key Features:**
  - **Brightness Extraction:** Calculates the average luminance of captured frames.
  - **Motion Detection:** Uses edge detection to assess motion intensity within the frame.

### Visualization

- **Class:** `NodeVisualizer`
- **Purpose:** Provides a separate window for visualizing the adaptive network in 3D space using Matplotlib.
- **Key Features:**
  - Real-time 3D scatter plot of node positions.
  - Dynamic updating to reflect network changes.

### Graphical User Interface

- **Class:** `App`
- **Purpose:** Serves as the main interface, integrating all components and facilitating user interactions.
- **Key Features:**
  - Control buttons for starting/stopping the system, accessing configurations, and managing visualizations.
  - Canvas for displaying live video feed with overlays indicating system movement and direction.
  - Menu options for saving and loading system states.

---

## Installation

### Prerequisites

- **Python 3.7 or higher**
- **Webcam:** Ensure that a functional webcam is connected to your system.

### INSTALL AND RUN 

```bash

git clone https://github.com/anttiluode/buginthemachine.git

cd buginthemachine

Create a Virtual Environment (Optional but Recommended)

python -m venv venv

source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt

pip install numpy opencv-python pillow matplotlib

python app.py
```

Main Window:

Start Button: Initiates the adaptive system, starting video capture and network processing.
Stop Button: Halts the adaptive system.
Visualize Nodes: Opens the 3D visualization window.
Config: Opens the configuration window to adjust system parameters.
Save: Saves the current system configuration and network state to a JSON file.
Load: Loads a saved system configuration and network state from a JSON file.
Video Canvas: Displays the live video feed with overlays indicating system position and direction.
Configuration
Access Configuration:

Click the Config button in the main window to open the configuration window.
Adjust Parameters:

Depth: Placeholder parameter for future enhancements.
Pruning Rate: Determines the threshold for pruning nodes (pruning_threshold).
Growth Rate: Probability of adding a new node at each interval (growth_rate).
Minimum Nodes: The lower bound on the number of nodes in the network.
Maximum Nodes: The upper bound on the number of nodes in the network.
Webcam Selection: Choose from available webcams connected to your system.
Save Configuration:

Use the Save Configuration button to export current settings and network state to a JSON file.
Load Configuration:

Use the Load Configuration button to import settings and network state from a JSON file.
Apply Changes:

After adjusting parameters, click the Apply button to enforce changes.
Visualization
Open 3D Visualization:

Click the Visualize Nodes button to open a separate window displaying the adaptive network in 3D.
Interacting with Visualization:

The visualization updates in real-time, reflecting changes in node positions and network structure.
Saving and Loading System States
Save System:

Click the Save button in the main window.
Choose a destination and filename to save the current configuration and network state as a JSON file.
Load System:

Click the Load button in the main window.
Select a previously saved JSON file to restore the system's configuration and network state.
Dependencies
Python Libraries:
NumPy: Numerical computations.
OpenCV: Computer vision and video processing.
Tkinter: Graphical user interface.
Pillow: Image processing.
Matplotlib: Data visualization.
Other Requirements:
Webcam: For capturing live video feed.
Troubleshooting
Webcam Access Issues:

Ensure that the webcam is properly connected and not being used by another application.
Verify that the correct webcam index is selected in the configuration.
Missing Dependencies:

If you encounter ModuleNotFoundError, ensure all dependencies are installed via pip install -r requirements.txt.
Performance Issues:

Running on systems with limited resources may result in slower performance.
Consider reducing the initial_nodes or adjusting growth_rate and pruning_threshold for better performance.
Visualization Window Not Opening:

Ensure that the main application is running and that the visualization window is not minimized or hidden behind other windows.
Error Messages:

Check the console or log output for detailed error messages.
Common issues include file I/O errors during saving/loading and webcam access failures.
Contributing
Contributions are welcome! Whether you're fixing bugs, improving documentation, or suggesting new features, your input is valuable.

Fork the Repository

Create a Feature Branch

bash
Copy code
git checkout -b feature/YourFeature
Commit Your Changes

bash
Copy code
git commit -m "Add your feature"
Push to the Branch

bash
Copy code
git push origin feature/YourFeature
Open a Pull Request

License
This project is licensed under the MIT License.

Acknowledgements
OpenCV: For providing powerful computer vision tools.
Tkinter Community: For maintaining the standard GUI library in Python.
NumPy and Matplotlib: For their indispensable roles in numerical computations and data visualization.
Python Developers: For creating an ecosystem that makes projects like "Bug in the Machine" possible.
