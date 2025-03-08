import matplotlib.pyplot as plt

def plot_two_frames(frame1, frame2, filepath: str):
    fig, ax = plt.subplots(1, 2, figsize=(6, 3))

    ax[0].imshow(frame1)
    ax[0].axis('off')
    ax[0].set_title("Frame 1")

    ax[1].imshow(frame2)
    ax[1].axis('off')
    ax[1].set_title("Frame 2")

    plt.tight_layout()
    plt.savefig(filepath, bbox_inches='tight', dpi=300)
    plt.close(fig)
