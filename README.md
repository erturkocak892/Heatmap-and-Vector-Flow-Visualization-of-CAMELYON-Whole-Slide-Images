# MM804 Project: Heatmap Visualization of CAMELYON Dataset

This project provides a full-stack web application for visualizing whole-slide images (WSI), running cell segmentation/classification inferences (via HoVer-Net), and visualizing the results through density heatmaps and vector overlays.

## Getting Started: Step-by-Step Guide

Follow these instructions carefully to get the project up and running. The codebase relies on an external repository (HoVer-Net) and its downloaded model weights to perform WSI inference correctly.

### Step 1: Clone this Repository
Clone this main project directory to your local machine:
```bash
git clone <YOUR_GITHUB_REPO_LINK>
cd "MM804 Project"
```

### Step 2: Download HoVer-Net and Required Model
To keep this repository lightweight, the HoVer-Net source code and its multi-gigabyte models are not included. The server uses a background process running HoVer-Net via Conda.

1. **Clone HoVer-Net Source Code:**
   Inside the root of this project, you must clone the `hover_net` repository. Ensure the folder is named **exactly** `hover_net`:
   ```bash
   git clone https://github.com/vqdang/hover_net.git hover_net
   ```

2. **Download the Pre-Trained Model Weights:**
   The specific model you need is the **Fast PanNuke (PyTorch) model**.
   - **Download Link**: Go to the [HoVer-Net Pretrained Models section](https://github.com/vqdang/hover_net#pretrained-models) (or directly via their provided Google Drive links in their README).
   - **File Name**: You are looking for a file named exactly: `hovernet_fast_pannuke_type_tf2pytorch.tar`.
   - **Where to put it**: Once downloaded, place this `.tar` file directly inside the `hover_net/` folder.
   
   Your directory structure should look like this:
   ```
   MM804 Project/
   ├── hover_net/
   │   ├── hovernet_fast_pannuke_type_tf2pytorch.tar   <-- Must be here!
   │   ├── environment.yml
   │   ├── requirements.txt
   │   └── ...
   ├── server/
   ├── web/
   └── Dockerfile
   ```

### Step 3: Add Sample WSI Data (622949.svs)
You will need our core test slide **`622949.svs`**.
1. **Download the Slide:** Download `622949.svs` from our shared Google Drive folder *(Author: Please insert the Google Drive link here)*.
2. **Place the Slide:** Move the downloaded `.svs` file into the `data/` directory.

```bash
mkdir -p data
# Place the slide here: data/622949.svs
```

### Step 4: Run the Application Using Docker (Recommended)
Because the codebase requires both a Python 3.10 environment (for FastAPI) and a Python 3.6 Conda environment (for HoVer-Net legacy dependencies), using Docker is by far the easiest way to launch the app safely without cluttering your system.

1. Install [Docker and Docker Compose](https://www.docker.com/products/docker-desktop/).
2. From the root of the project, run:
   ```bash
   docker-compose up --build
   ```
   *(Note: The first time this is run, it will take several minutes to download Miniconda, create the `hovernet` conda environment, and install all deep learning packages).*

### Step 5: Test the WSI Pipeline
To verify everything is working:
1. Open your browser and go to [http://localhost:8000/](http://localhost:8000/).
2. You will see the application interface.
3. Select an existing slide or upload a new one via the "Upload Slide" button.
4. **(Note: The Draw ROI polygon tool is currently under maintenance.** Instead of drawing a polygon, simply zoom in/pan on the viewport to your desired tissue area).
5. Open the **Analyze** tab and click **"Run Inference"**. This will automatically process the region currently visible on your screen.
6. You will see a progress bar. The server is extracting the patch and running `hover_net` in the background.
7. Once completed, toggle the **"Inference Results"** or **"Cell Density Heatmap"** overlays. You should clearly see the cellular segmentation contours and hot/cold density mapping.

---

## Running Locally (Without Docker)

If you must run the application natively on your OS, perform these steps:

**1. Install System level Dependencies:**
- **macOS (via Homebrew):** `brew install openslide`
- **Ubuntu/Debian:** `sudo apt-get install openslide-tools libgl1 libglib2.0-0`

**2. Setup the `hovernet` Conda Environment (For Inference):**
Ensure you have Conda/Miniconda installed.
```bash
cd hover_net
conda env create -f environment.yml
cd ..
```

**3. Setup the Main FastApi Environment (For Web Server):**
Create a separate standard virtual environment (Python 3.10+):
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**4. Start the Application Server:**
With your `venv` active, run:
```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
```
Access the application at `http://localhost:8000/`.
