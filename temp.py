from transformers import AutoTokenizer, CLIPTextModel

model_id = "openai/clip-vit-base-patch32"
save_dir = "./pretrained/clip-vit-base-patch32"

AutoTokenizer.from_pretrained(model_id).save_pretrained(save_dir)
CLIPTextModel.from_pretrained(model_id).save_pretrained(save_dir)
