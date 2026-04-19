import kagglehub

path = kagglehub.dataset_download(
    "bmshahriaalam/tealeafbd-tea-leaf-disease-detection"
)

print("Path:", path)
