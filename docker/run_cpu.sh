docker run -it --rm -p 3000:3000 --name sbv2 \
-v ./models:/work/models --env-file .env \
ghcr.io/tuna2134/sbv2-api:cpu