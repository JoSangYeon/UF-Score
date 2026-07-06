import torch
from PIL import Image
from transformers import AutoProcessor, Blip2ForImageTextRetrieval, Blip2ForConditionalGeneration
import pandas as pd
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

model = Blip2ForImageTextRetrieval.from_pretrained("Salesforce/blip2-itm-vit-g", torch_dtype=torch.float16)
model.to(device)
processor = AutoProcessor.from_pretrained("Salesforce/blip2-itm-vit-g")

csv_datas = ["Valor_32k_train.csv", "VGG_200k_train.csv"]
for csv_data in csv_datas:
    valor_data = pd.read_csv(csv_data)
    if "Valor_32k_train" in csv_data:
        data_root_path = "/mnt/storage/ufscore/workspace/_DATASET/VALOR-32K"
    else:
        data_root_path = "/mnt/storage/ufscore/workspace/_DATASET/vgg_sound"

    result = []
    for idx, row in valor_data.iterrows():
        img_path = os.path.join(data_root_path, row["img_path"])
        image = Image.open(img_path)
        inputs = processor(images=image, text=row["caption"], return_tensors="pt").to(device, torch.float16)
        with torch.no_grad():
            itc_out = model(**inputs, use_image_text_matching_head=False)
            logits_per_image = itc_out.logits_per_image  # this is the image-text similarity score
            
        result.append({
            "img_path": row["img_path"],
            "wav_path": row["wav_path"],
            "caption": row["caption"],
            "BLIP2_score": logits_per_image.detach().item(),
        })
        if idx % 500 == 0:
            print(f"Progress: {idx}/{len(valor_data)}")

    result = pd.DataFrame(result)
    result.to_csv(f"{csv_data.replace('.csv', '_blip2_score.csv')}", index=False)


