import matplotlib.pyplot as plt
import seaborn as sns

def generate_player_visualizations(df_primary_player, primary_player_adjusted_summary, primary_player_track_id):
    """
    Generates and displays visualizations for the primary player's movement data.

    Args:
        df_primary_player (pd.DataFrame): DataFrame containing the primary player's detailed movement data.
        primary_player_adjusted_summary (pd.DataFrame): Summary DataFrame of the primary player's adjusted metrics.
        primary_player_track_id (int): The track ID of the primary player.
    """
    print(f"\n--- Generating Visualizations for Primary Player (Track ID: {int(primary_player_track_id)}) ---")

    # 1. Visualize Primary Player Trajectory
    plt.figure(figsize=(10, 8))
    sns.lineplot(data=df_primary_player, x='x_center', y='y_center', hue='frame_id', palette='viridis', legend=False)
    plt.title(f'Primary Player Trajectory (Track ID: {int(primary_player_track_id)})')
    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')
    plt.grid(True)
    plt.gca().invert_yaxis() # Invert y-axis to match typical video coordinates (origin top-left)
    plt.show()

    # 2. Create a Heatmap of Primary Player Activity
    plt.figure(figsize=(10, 8))
    plt.hist2d(df_primary_player['x_center'], df_primary_player['y_center'], bins=50, cmap='hot_r', cmin=1)
    plt.colorbar(label='Activity Count')
    plt.title(f'Heatmap of Primary Player Activity (Track ID: {int(primary_player_track_id)})')
    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')
    plt.gca().invert_yaxis() # Invert y-axis to match typical video coordinates (origin top-left)
    plt.grid(True)
    plt.show()

    # 3. Visualize Primary Player Total Adjusted Distance
    plt.figure(figsize=(8, 6))
    sns.barplot(x='track_id', y='total_adjusted_distance_moved', data=primary_player_adjusted_summary, hue='track_id', palette='viridis', legend=False)
    plt.title(f'Primary Player Total Adjusted Distance Moved (Track ID: {int(primary_player_track_id)})')
    plt.xlabel('Player ID')
    plt.ylabel('Total Adjusted Distance Moved (pixels)')
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # 4. Visualize Primary Player Average Adjusted Speed
    plt.figure(figsize=(8, 6))
    sns.barplot(x='track_id', y='average_adjusted_speed', data=primary_player_adjusted_summary, hue='track_id', palette='mako', legend=False)
    plt.title(f'Primary Player Average Adjusted Speed (Track ID: {int(primary_player_track_id)})')
    plt.xlabel('Player ID')
    plt.ylabel('Average Adjusted Speed (pixels/frame)')
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    print("Visualizations for primary player's adjusted movement data generated.")
