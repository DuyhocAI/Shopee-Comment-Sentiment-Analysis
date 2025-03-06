import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import json

# Kiểm tra GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Đường dẫn model đã lưu
model_path = r"D:\Shopee\Data_expl\fine_tuned_sentiment_classifier.pth"
new_dataset_path = r"D:\Shopee\Data_expl\new_sentiment_dataset_1.json"

# Tokenizer (PhoBERT)
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

# Mô hình
class SentimentClassifier(nn.Module):
    def __init__(self, pretrained_model_name, num_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(pretrained_model_name)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        logits = self.fc(cls_output)
        return logits

# Khởi tạo mô hình và tải trọng số
num_classes = 3  # Negative, Neutral, Positive
model = SentimentClassifier(pretrained_model_name="vinai/phobert-base", num_classes=num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# Hàm dự đoán cảm xúc
def predict_sentiment(text, model, tokenizer, device):
    model.eval()
    with torch.no_grad():
        encoding = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, prediction = torch.max(outputs, dim=1)
    return prediction.item()

# Map nhãn
label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

# Ví dụ kiểm tra và lưu kết quả
new_dataset = []
while True:
    text = input("Enter a comment (or 'exit' to quit): ")
    if text.lower() == 'exit':
        break

    predicted_label = predict_sentiment(text, model, tokenizer, device)
    print(f"Predicted Sentiment: {label_map[predicted_label]}")

    # Đánh giá đúng/sai từ người dùng
    is_correct = input("Is this prediction correct? (yes/no): ").strip().lower()
    correct_label = predicted_label if is_correct == 'yes' else int(input("Enter the correct label (0: Negative, 1: Neutral, 2: Positive): "))

    # Lưu vào dataset mới
    new_dataset.append({
        "text": text,
        "predicted_label": label_map[predicted_label],
        "correct_label": label_map[correct_label]
    })

# Ghi dataset mới vào file JSON
with open(new_dataset_path, 'w', encoding='utf-8') as f:
    json.dump(new_dataset, f, ensure_ascii=False, indent=4)

print(f"New dataset saved to {new_dataset_path}")
