# 1. build + start
docker compose up --build

# 2. sanity checks
curl localhost:8000/health
# -> {"status":"ok"}

curl -X POST localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"text":"hello there"}'
# -> {"probability":0.873}
