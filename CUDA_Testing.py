import torch as hhhhhh
print(hhhhhh.cuda.is_available())  # Should return True


# Create a tensor and move it to GPU
x = hhhhhh.rand(5, 5).cuda()
print(x)
