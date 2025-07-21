import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def visualize_frame_with_chairs(frame_path, chair_data):
    """Visualize a frame with tracked chairs, highlighting occupied ones"""
    img = Image.open(frame_path)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img)
    
    for chair in chair_data:
        chair_id = chair['id']
        x1, y1, x2, y2 = chair['bbox']
        is_occupied = chair.get('occupied', False)
        
        # Choose color based on occupancy
        color = 'red' if is_occupied else 'cyan'
        label = f"Chair-{chair_id} {'(Occupied)' if is_occupied else ''}"
        
        # Create rectangle
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)
        
        # Add ID text
        plt.text(
            x1, y1 - 10, label,
            color=color, fontsize=12, weight='bold'
        )
    
    plt.axis('off')
    plt.title(f'Tracked Chairs (Red = Occupied)')
    plt.tight_layout()
    return fig

def plot_detected_vs_occupied(analysis_results):
    """
    Plot detected chairs vs occupied chairs over time
    
    Args:
        analysis_results: Dictionary of analysis results
    """
    # Extract frame numbers and sort them
    frame_numbers = sorted(analysis_results.keys())
    
    # Extract data for plotting
    detected_chairs = [analysis_results[f]['chairs'] for f in frame_numbers]
    occupied_chairs = [analysis_results[f]['seated'] for f in frame_numbers]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot both lines
    ax.plot(frame_numbers, detected_chairs, 'b-', linewidth=2, label='Detected Chairs')
    ax.plot(frame_numbers, occupied_chairs, 'r-', linewidth=2, label='Occupied Chairs')
    
    # Add fill between the lines
    ax.fill_between(frame_numbers, detected_chairs, occupied_chairs, 
                    where=[d >= o for d, o in zip(detected_chairs, occupied_chairs)],
                    facecolor='green', alpha=0.3, interpolate=True)
    
    # Formatting
    ax.set_xlabel('Frame Number', fontsize=12)
    ax.set_ylabel('Chair Count', fontsize=12)
    ax.set_title('Detected Chairs vs Occupied Chairs Over Time', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add utilization percentage
    if detected_chairs:
        utilization = sum(occupied_chairs) / sum(detected_chairs) * 100
        ax.text(0.05, 0.95, f"Avg Utilization: {utilization:.1f}%", 
                transform=ax.transAxes, fontsize=12,
                bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig