# Photo/Text to 3D Model Prototype

## Overview

This prototype converts either a **text prompt** (e.g. "a small toy car") or an **image** (e.g. photo of a chair) into a basic 3D model in `.ply` format. It uses OpenAI’s Point-E library for 3D generation and supports basic visualization.

## Setup

```bash
python -m venv venv
source venv/bin/activate  # or venv\\Scripts\\activate on Windows
pip install -r requirements.txt