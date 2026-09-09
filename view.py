import pickle
import matplotlib.pyplot as plt

with open("predictions.pkl", "rb") as f:
    images_data = pickle.load(f)


def plot_results(images_data):
    
    # initial state -> start with the first image 
    state = {"index": 0}
    
    # create a figure with 3 subplots: original image, relevance map, and top predictions
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), gridspec_kw={'width_ratios': [1, 1, 1.2]})
    
    def draw():
        index = state["index"]
        image_data = images_data[index]
        
        for ax in axes:
            ax.clear()
            
        # plot original image
        axes[0].imshow(image_data["original_image"])
        axes[0].set_title(f"[{index+1}/{len(images_data)}]  {image_data['path']}")
        axes[0].axis("off")
        
        # plot relevance map
       # axes[1].imshow(image_data["original_image"])
        axes[1].imshow(image_data["relevance_map"], cmap="hot") # seismic, alpha 0.5
        axes[1].set_title("Relevance Map")
        axes[1].axis("off")
        
        # plot top predictions
        y_pos = range(len(image_data["top_classes"]))
        axes[2].barh(y_pos, image_data["top_probabilities"][::-1])
        axes[2].set_yticks(y_pos)
        axes[2].set_yticklabels(image_data["top_classes"][::-1], fontsize=8)
        axes[2].set_xlabel("Probability")
        axes[2].set_xlim(0, 1)
        axes[2].set_title("Top Predictions")
        
        fig.tight_layout()
        fig.canvas.draw_idle()
        
    def on_key(event):
        if event.key == " ":
            state["index"] = (state["index"] + 1) % len(images_data)
            draw()
        elif event.key == "backspace":
            state["index"] = (state["index"] - 1) % len(images_data)
            draw()
                
    fig.canvas.mpl_connect("key_press_event", on_key)
    draw()
    plt.show()
        
    
    
def main():
    plot_results(images_data)
    
if __name__ == "__main__":
    main()