DEVICE = cpu
PORT = 8888
GPU = all

# Set GPU_FLAG based on DEVICE
GPU_FLAG = $(if $(filter-out cpu,$(DEVICE)),--gpus=$(GPU),)

install:
	pip install -r requirements-$(DEVICE).txt

build_image:
	docker build -t representation_learning -f $(DEVICE).dockerfile .

run_container:
	docker run --rm $(GPU_FLAG) -p $(PORT):8888 -v `pwd`:/assignment --name representation_learning representation_learning
