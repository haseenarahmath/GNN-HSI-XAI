# create folders
mkdir -p eegnn-xai/src/{core,data,models,explainers} results

# save the files above into those paths

# install deps
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# train on synthetic data
python main.py --mode train --dataset synthetic --epochs 3

# explain with IG / Saliency / GradCAM
python main.py --mode explain --dataset synthetic --explainer ig --topk 5
python main.py --mode explain --dataset synthetic --explainer saliency --topk 5
python main.py --mode explain --dataset synthetic --explainer gradcam --topk 5
