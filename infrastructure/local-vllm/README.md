# vLLM local avec GPU NVIDIA

Cette configuration expose une API compatible OpenAI uniquement sur
`http://127.0.0.1:8000/v1`. Elle est destinée au développement local et ne
remplace pas la configuration Scaleway de production.

## Prérequis

- GPU NVIDIA compatible CUDA et pilote NVIDIA fonctionnel ;
- Docker, le plugin Docker Compose v2 et NVIDIA Container Toolkit ;
- environ 8 Go de VRAM pour le modèle par défaut.

Sous Ubuntu, installer le plugin Compose manquant :

```bash
sudo apt update
sudo apt install docker-compose-v2
docker compose version
```

Vérifier ensuite que la machine et Docker voient le GPU :

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

Si `lspci | grep -Ei 'nvidia|vga|3d'` ne montre aucun GPU NVIDIA, cette
configuration vLLM CUDA ne peut pas fonctionner sur la machine.

## Démarrage

Depuis `infrastructure/local-vllm` :

```bash
cp .env.example .env
docker compose --env-file .env config
docker compose --env-file .env up -d
docker compose --env-file .env logs -f vllm
```

Le premier démarrage télécharge plusieurs gigaoctets. Quand le conteneur est
prêt, vérifier le modèle servi :

```bash
curl http://127.0.0.1:8000/v1/models
```

L'identifiant attendu est `restaurant-assistant`.

## Voice Server

Configurer `.env` sans clé OpenAI :

```dotenv
LLM_PROVIDER=vllm
VLLM_BASE_URL=http://127.0.0.1:8000/v1
VLLM_MODEL=restaurant-assistant
VLLM_API_KEY=
LLM_FALLBACK_GPT=false
OPENAI_API_KEY=
```

Configurer également le backend local avec `VOICE_PROVIDER=python`. Fastify
enverra alors le flux vers `ws://127.0.0.1:8090/media-stream` au lieu
d'ouvrir une session OpenAI Realtime.

## Test d'intégration local

Depuis la racine du dépôt `voice-server`, après installation des dépendances
de développement :

```bash
export RUN_LOCAL_VLLM_TESTS=1
export VLLM_BASE_URL=http://127.0.0.1:8000/v1
export VLLM_MODEL=restaurant-assistant
pytest tests/test_local_vllm_e2e.py -v
```

Ce test simule le WebSocket et l'audio Twilio, utilise réellement vLLM, puis
vérifie qu'une réponse audio repart vers Twilio. Le VAD, le STT et le TTS sont
remplacés par des doubles déterministes afin que le test cible l'intégration
vLLM et non les modèles audio.

## Arrêt

```bash
docker compose --env-file .env down
```
