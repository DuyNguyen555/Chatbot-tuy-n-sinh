from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from duckduckgo_search import DDGS
from sentence_transformers import SentenceTransformer, util
import json
import torch
from model.model import NeuralNet
from model.nltk_utils import bag_of_words, tokenize
import random

def index(request):
    return render(request, 'home/index.html')

# Load model và dữ liệu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('model/data.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

FILE = 'model/data.pth'
data = torch.load(FILE)

input_size = data['input_size']
hidden_size = data['hidden_size']
output_size = data['output_size']
all_words = data['all_words']
tags = data['tags']
model_state = data['model_state']

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

@csrf_exempt
def chatbot_response(request):
    if request.method == 'POST':
        data_req = json.loads(request.body)
        sentence = data_req['message']

        sentence_token = tokenize(sentence)
        X = bag_of_words(sentence_token, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device)

        output = model(X)
        _, predicted = torch.max(output, dim=1)
        tag = tags[predicted.item()]
        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]

        if prob.item() > 0.75:
            for intent in intents['intents']:
                if tag == intent['tag']:
                    response = random.choice(intent['responses'])
                    return JsonResponse({'response': response})
        else:
            return JsonResponse({'response': "Tôi không hiểu lắm"})
            # # Dùng mô hình tiếng Việt
            # model_v = SentenceTransformer("bkai-foundation-models/vietnamese-bi-encoder")

            # def search_link(query, site_filter="tuyensinhso.vn", max_results=10):
            #     # Tạo đối tượng DDGS và thực hiện tìm kiếm
            #     with DDGS() as ddgs:
            #         # Tìm kiếm với từ khóa, site filter và số lượng kết quả tối đa
            #         results = ddgs.text(
            #             keywords=site_filter + " " + query,  # Kết hợp site filter và query
            #             max_results=max_results  # Giới hạn số lượng kết quả
            #         )
                    
            #     return results

            # # Ví dụ tìm kiếm
            # # query = input("Bạn muốn tìm gì?\nBạn: ")
            # result = search_link(request)
            # research_result = [result[text]['body'] for text in range(len(result))]

            # query_emb = model_v.encode(request, convert_to_tensor=True)
            # result_embs = model_v.encode(research_result, convert_to_tensor=True)

            # cos_scores = util.pytorch_cos_sim(query_emb, result_embs)[0]

            # top_k = 3
            # top_results = sorted(zip(research_result, cos_scores), key=lambda x: x[1], reverse=True)[:top_k]
            # return JsonResponse({'response': top_results[0][0]})