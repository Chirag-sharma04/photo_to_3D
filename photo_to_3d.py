import os
import argparse
import numpy as np
import cv2
import torch
import trimesh
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import cm
from diffusers import StableDiffusionPipeline, DiffusionPipeline
from transformers import CLIPProcessor, CLIPModel

# For visualization
import pyrender

class PhotoText3DConverter:
    def __init__(self):
        """Initialize the converter with necessary models."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load CLIP for image understanding
        self.clip_model = None
        self.clip_processor = None
        
        # Load text-to-image model for visualization
        self.text_to_image_model = None
        
        # Initialize depth estimation model
        self.depth_model = None
        
    def load_models(self):
        """Load all required models."""
        print("Loading models...")
        
        # Load CLIP model for image understanding
        if self.clip_model is None:
            print("Loading CLIP model...")
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_model.to(self.device)
        
        # Initialize MiDaS depth estimation
        if self.depth_model is None:
            print("Loading depth estimation model...")
            self.depth_model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
            self.depth_model.to(self.device)
            self.depth_model.eval()
            
        # Load text-to-image model for text inputs
        if self.text_to_image_model is None and self.device == "cuda":
            print("Loading text-to-image model...")
            self.text_to_image_model = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5", 
                torch_dtype=torch.float16
            )
            self.text_to_image_model.to(self.device)
        
        print("Models loaded.")
        
    def process_input(self, input_path, is_text=False):
        """Process either an image path or a text prompt."""
        if is_text:
            return self._process_text(input_path)
        else:
            return self._process_image(input_path)
            
    def _process_image(self, image_path):
        """Process an image by extracting depth information."""
        print(f"Processing image: {image_path}")
        
        # Read and preprocess the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image at {image_path}")
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get the depth map
        depth_map = self._extract_depth_map(image)
        
        # Create 3D mesh from depth map
        mesh = self._depth_to_mesh(depth_map, image)
        
        return mesh, image, depth_map
    
    def _process_text(self, text_prompt):
        """Process a text prompt by generating an image first."""
        print(f"Processing text prompt: {text_prompt}")
        
        # Generate an image from text if model is available
        if self.text_to_image_model is not None and self.device == "cuda":
            with torch.no_grad():
                image = self.text_to_image_model(text_prompt).images[0]
                # Convert from PIL to numpy
                image_np = np.array(image)
        else:
            # Create a simple placeholder image with text
            image_np = np.ones((512, 512, 3), dtype=np.uint8) * 255
            # Add text to image
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image_np, text_prompt, (50, 250), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
        
        # Get the depth map
        depth_map = self._extract_depth_map(image_np)
        
        # Create 3D mesh from depth map
        mesh = self._depth_to_mesh(depth_map, image_np)
        
        return mesh, image_np, depth_map
    
    def _extract_depth_map(self, image):
        """Extract depth map from an image using MiDaS."""
        # Resize image for the model
        input_batch = self._prepare_depth_input(image)
        
        # Compute depth
        with torch.no_grad():
            prediction = self.depth_model(input_batch)
            
            # Resize to original size
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=image.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
            
        depth_map = prediction.cpu().numpy()
        
        # Normalize depth map
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
        
        return depth_map
    
    def _prepare_depth_input(self, img):
        """Prepare image for depth estimation model."""
        # Resize to model input size
        img_input = cv2.resize(img, (384, 384))
        
        # Convert to torch tensor and normalize
        img_input = torch.from_numpy(img_input).permute(2, 0, 1).float() / 255.0
        img_input = img_input.unsqueeze(0).to(self.device)
        
        return img_input
    
    def _depth_to_mesh(self, depth_map, color_image=None):
        """Convert depth map to a 3D mesh."""
        # Get dimensions
        h, w = depth_map.shape
        
        # Create grid of vertices
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        
        # Create vertices
        vertices = np.zeros((h, w, 3))
        vertices[:, :, 0] = x
        vertices[:, :, 1] = y
        vertices[:, :, 2] = depth_map * 10  # Scale depth for better visualization
        
        # Reshape vertices
        vertices = vertices.reshape(-1, 3)
        
        # Create faces (triangles)
        faces = []
        for i in range(h - 1):
            for j in range(w - 1):
                v0 = i * w + j
                v1 = i * w + j + 1
                v2 = (i + 1) * w + j
                v3 = (i + 1) * w + j + 1
                
                # Create two triangles per grid cell
                faces.append([v0, v1, v2])
                faces.append([v1, v3, v2])
        
        faces = np.array(faces)
        
        # Create trimesh object
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        
        # Add vertex colors if color image is provided
        if color_image is not None:
            # Reshape the color image to match vertices
            vertex_colors = color_image.reshape(-1, 3)
            
            # Scale to 0-1 range
            vertex_colors = vertex_colors / 255.0
            
            # Apply color to mesh
            mesh.visual.vertex_colors = vertex_colors
            
        return mesh
    
    def save_mesh(self, mesh, output_path, format="obj"):
        """Save the mesh to a file."""
        if format == "obj":
            mesh.export(output_path)
        elif format == "stl":
            mesh.export(output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
            
        print(f"Mesh saved to {output_path}")
        
    def visualize_results(self, image, depth_map, mesh, output_dir):
        """Visualize the results."""
        fig = plt.figure(figsize=(15, 5))
        
        # Original image
        ax1 = fig.add_subplot(131)
        ax1.imshow(image)
        ax1.set_title("Input Image")
        ax1.axis("off")
        
        # Depth map
        ax2 = fig.add_subplot(132)
        ax2.imshow(depth_map, cmap="viridis")
        ax2.set_title("Depth Map")
        ax2.axis("off")
        
        # 3D visualization using matplotlib
        ax3 = fig.add_subplot(133, projection="3d")
        
        # Downsample for faster visualization
        h, w = depth_map.shape
        step = 5  # Adjust based on image size
        
        y, x = np.mgrid[0:h:step, 0:w:step]
        z = depth_map[0:h:step, 0:w:step] * 10  # Scale depth
        
        surf = ax3.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        ax3.set_title("3D Surface")
        
        # Save the figure
        plt.savefig(os.path.join(output_dir, "visualization.png"))
        print(f"Visualization saved to {os.path.join(output_dir, 'visualization.png')}")
        
        # Return the figure for optional display
        return fig
    
    def process(self, input_path, output_dir, is_text=False, format="obj"):
        """Process input and generate 3D model."""
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load models if not already loaded
        self.load_models()
        
        # Process input
        mesh, image, depth_map = self.process_input(input_path, is_text)
        
        # Create output path
        if is_text:
            # Sanitize text for filename
            input_name = "".join([c if c.isalnum() else "_" for c in input_path])
            output_path = os.path.join(output_dir, f"{input_name[:30]}.{format}")
        else:
            # Get filename without extension
            input_name = os.path.splitext(os.path.basename(input_path))[0]
            output_path = os.path.join(output_dir, f"{input_name}.{format}")
        
        # Save mesh
        self.save_mesh(mesh, output_path, format=format)
        
        # Visualize results
        fig = self.visualize_results(image, depth_map, mesh, output_dir)
        
        return output_path, fig

def main():
    """Main function to handle command-line arguments."""
    parser = argparse.ArgumentParser(description="Convert photo or text prompt to 3D model")
    parser.add_argument("--input", required=True, help="Path to input image or text prompt")
    parser.add_argument("--output", default="output", help="Directory to save output files")
    parser.add_argument("--text", action="store_true", help="Treat input as text prompt")
    parser.add_argument("--format", choices=["obj", "stl"], default="obj", help="Output format")
    
    args = parser.parse_args()
    
    # Create converter
    converter = PhotoText3DConverter()
    
    # Process input
    output_path, _ = converter.process(args.input, args.output, args.text, args.format)
    
    print(f"3D model saved to: {output_path}")

if __name__ == "__main__":
    main()