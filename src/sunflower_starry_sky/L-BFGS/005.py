
from pathlib import Path
import csv

import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageOps
from torchvision import transforms
from torchvision.models import VGG19_Weights, vgg19

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print("Using device:", device)

STYLE_LAYERS = [1, 6, 11, 20, 29]
CONTENT_LAYER = 22

ALPHA = 1.0
BETA = 100.0
GAMMA = 50.0

LEARNING_RATE = 1.0
MAX_ITER = 1
HISTORY_SIZE = 10

NUM_STEPS = 150
SAVE_EVERY = 30

IMAGE_MAX_SIZE = 1024

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)

def load_image(path):
    image = Image.open(path).convert("RGB")
    image = ImageOps.contain(image, (IMAGE_MAX_SIZE, IMAGE_MAX_SIZE))
    return transforms.ToTensor()(image).unsqueeze(0)

def extract_features(image_tensor, network):
    current = normalize(image_tensor)
    selected = {}
    for layer_index, layer in enumerate(network):
        current = layer(current)
        if layer_index in STYLE_LAYERS or layer_index == CONTENT_LAYER:
            selected[layer_index] = current
        if layer_index == 29:
            break
    return selected

def gram_matrix(feature):
    _, channels, height, width = feature.shape
    matrix = feature[0].reshape(channels, height * width)
    return (matrix @ matrix.T) / (height * width)

def total_variation_loss(image):
    horizontal = (image[:, :, :, 1:] - image[:, :, :, :-1])
    vertical = (image[:, :, 1:, :] - image[:, :, :-1, :])

    return ((horizontal ** 2).mean() + (vertical ** 2).mean())

def save_image(image_tensor, filename):
    pil_image = transforms.ToPILImage()(
        image_tensor.detach().cpu().squeeze(0)
    )
    pil_image.save(filename)

output_dir = Path("outputs/sunflower_starry_sky/L-BFGS/5")
output_dir.mkdir(parents=True, exist_ok=True)

content = load_image("sunflower_content.jpg").to(device)
style = load_image("starry_sky_style.jpg").to(device)

network = (vgg19(weights=VGG19_Weights.DEFAULT).features.eval().requires_grad_(False).to(device))

with torch.no_grad():
    content_features = extract_features(content, network)
    style_features = extract_features(style, network)
    content_target = (content_features[CONTENT_LAYER].detach())
    style_targets = {}
    for layer_index in STYLE_LAYERS:
        style_targets[layer_index] = gram_matrix(
            style_features[layer_index]
        ).detach()

generated = content.clone().requires_grad_(True)

optimizer = torch.optim.LBFGS(
    params=[generated],
    lr=LEARNING_RATE,
    max_iter=MAX_ITER,
    history_size=HISTORY_SIZE,
    line_search_fn="strong_wolfe",
)

def calculate_losses():
    generated_features = extract_features(generated, network)

    content_loss = (
            (generated_features[CONTENT_LAYER] - content_target) ** 2
        ).mean()

    style_loss = 0.0
    for layer_index in STYLE_LAYERS:
        generated_gram = gram_matrix(generated_features[layer_index])
        style_loss = style_loss + (
            (
                generated_gram
                - style_targets[layer_index]
            ) ** 2
        ).mean()
    style_loss = style_loss / len(STYLE_LAYERS)

    tv_loss = total_variation_loss(generated)
    total_loss = (
        ALPHA * content_loss
        + BETA * style_loss
        + GAMMA * tv_loss
    )

    return (
        content_loss,
        style_loss,
        tv_loss,
        total_loss
    )

closure_calls = [0]

def closure():
    closure_calls[0] += 1
    optimizer.zero_grad()

    _, _, _, total_loss = calculate_losses()

    total_loss.backward()
    
    return total_loss

steps = []

content_values = []
style_values = []
tv_values = []
total_values = []

weighted_content_values = []
weighted_style_values = []
weighted_tv_values = []

closure_evaluation_values = []

for step in range(NUM_STEPS + 1):

    with torch.no_grad():
        (content_loss, style_loss, tv_loss, total_loss) = calculate_losses()

    content_value = content_loss.item()
    style_value = style_loss.item()
    tv_value = tv_loss.item()
    total_value = total_loss.item()

    steps.append(step)

    content_values.append(content_value)
    style_values.append(style_value)
    tv_values.append(tv_value)
    total_values.append(total_value)
    weighted_content_values.append( ALPHA * content_value)
    weighted_style_values.append(BETA * style_value)
    weighted_tv_values.append(GAMMA * tv_value)
    closure_evaluation_values.append(closure_calls[0])

    print(
        f"Step calls {step:03d} | "
        f"content={content_value:.6f} "
        f"style={style_value:.6f} "
        f"tv={tv_value:.6f} "
        f"weighted_total={total_value:.6f} "
        f"closure_evaluations={closure_calls[0]}"
    )

    if step % SAVE_EVERY == 0 or step == NUM_STEPS:
        image_path = (output_dir / f"step_{step:03d}.png")
        save_image(generated, image_path)
        print("Saved image:", image_path)

    if step == NUM_STEPS:
        break

    optimizer.step(closure)
    with torch.no_grad():
        generated.clamp_(0.0, 1.0)

csv_path = output_dir / "loss_history.csv"

with csv_path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow([
        "updates_completed","content_raw", "style_raw", "tv_raw",
        "content_weighted", "style_weighted", "tv_weighted","total_weighted", "closure_evaluations_total",
    ])
    for i in range(len(steps)):
        writer.writerow([
            steps[i], content_values[i], style_values[i], tv_values[i],
            weighted_content_values[i], weighted_style_values[i], weighted_tv_values[i], total_values[i],
            closure_evaluation_values[i],
        ])

plt.figure(figsize=(8, 5))
plt.plot(steps, weighted_content_values, label="Content (weighted)")
plt.plot(steps, weighted_style_values, label="Style (weighted)")
plt.plot(steps, weighted_tv_values, label="TV (weighted)")
plt.plot(steps, total_values,label="Total (weighted)")
plt.xlabel("Completed L-BFGS step() calls")
plt.ylabel("Loss value")
plt.title("Neural Style Transfer Losses (L-BFGS + Strong Wolfe)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plot_path = output_dir / "loss_curves.png"
plt.savefig(plot_path, dpi=160)
plt.close()

print("Saved loss records:", csv_path)
print("Saved loss curves:", plot_path)

print("Final image:",output_dir / f"step_{NUM_STEPS:03d}.png")
print("Total closure evaluations:",closure_calls[0])