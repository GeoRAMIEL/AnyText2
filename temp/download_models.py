from modelscope import snapshot_download
your_path_to_model_dir = snapshot_download('iic/cv_anytext2')
print("models downloaded to:", your_path_to_model_dir)
