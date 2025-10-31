import torch
sd = torch.load("models/cnn1d_bin.pt", map_location="cpu")
# dict 래핑일 경우 풀기 (plot_eval.py의 _load_torch_state와 동일 로직)
sd = sd.get("model_state", sd.get("state_dict", sd))

print(sd["conv1.weight"].shape)  #  torch.Size([32, 1, 3])
print(sd["conv2.weight"].shape)  #  torch.Size([64, 32, 3])
print(sd["fc.weight"].shape)     #  torch.Size([1, 64])