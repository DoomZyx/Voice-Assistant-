# Architecture — Voice Service V1

Service vocal **local, modulaire et independant**. Fastify reste le backend metier (reservations, commandes, pricing, auth). Le LLM n'a **aucun acces direct a la base**. Toutes les actions passent par le Tool Dispatcher, qui appelle Fastify en HTTP.

## Fallback GPT

Deux niveaux, toujours prefixe **`FALLBACK GPT:`** dans les logs :

1. **Appel entier (Fastify)** — Twilio reste sur `GET /media-stream` Fastify. Fastify sonde `VOICE_HEALTH_URL`. Si le Voice Service est injoignable ou `engines.ready` est faux, Fastify branche OpenAI Realtime (`path: openai-realtime`).
2. **LLM seulement (Voice Service)** — STT/TTS locaux restent actifs. Si vLLM refuse une completion et `LLM_FALLBACK_GPT=true` + `OPENAI_API_KEY`, bascule OpenAI Chat Completions (`gpt-4o-mini` par defaut).

Ne pas pointer `VOICE_STREAM_URL` directement sur le port Python : le fallback Realtime de Fastify ne s'executerait plus.

## Flux V1

```
Twilio (appel)
    |  webhook TwiML (Fastify /incoming-call)
    |  <Stream url="wss://.../media-stream">
    v
Twilio Media Streams (mulaw 8 kHz)
    v
Voice Service FastAPI  (/media-stream)
    |  Session isolee (buffer audio + historique + cancel)
    v
VAD Silero (ONNX)
    |  fin de tour de parole
    v
STT faster-whisper  (utterance complete)
    v
LLM vLLM  (/v1/chat/completions, tools)
    |
    +-- tool_calls --> Tool Dispatcher --HTTP--> Fastify
    |                    (check_availability / create_appointment)
    |                         |
    |                         v
    |                    Fastify (SQL/Mongo, droits, validation)
    |                         |
    +<-- JSON resultat -------+
    |
    v
TTS Kokoro  (PCM 24 kHz -> PCM 8 kHz -> mulaw)
    v
Twilio Media Streams  (event media + clear si barge-in)
```

## Composants

| Module | Role | Isolation |
|---|---|---|
| `app/twilio` | Protocole Media Streams (start/media/stop, clear) | 1 WebSocket = 1 appel |
| `app/session` | Session, buffer, orchestration non-bloquante | 1 `CallSession` par WS |
| `app/vad` | Silero ONNX, etat RNN **par session** | Modele partage, etat prive |
| `app/stt` | faster-whisper | Modele partage, lock asyncio |
| `app/llm` | Interface interchangeable. V1 = vLLM | HTTP async |
| `app/tools` | Dispatcher vers Fastify uniquement | Pas de DB |
| `app/tts` | Kokoro | Modele partage, lock asyncio |
| Fastify | Metier, validation, persistances | Inchange |

Le receive-loop WebSocket ne bloque pas sur STT / LLM / TTS : un tour de parole est un `asyncio.Task`. Les sessions ne partagent pas l'historique ni le buffer.

## Contrat Fastify (inchange)

Le Voice Service appelle uniquement :

- `GET /api/ping`
- `GET /api/orders/ai/available-slots?date=`
- `GET /api/reservations/ai/available-slots?date=`
- `POST /api/reservations/ai/create`
- `POST /api/orders/ai/create`

Header optionnel : `x-api-key` (`NODE_API_KEY`).

Twilio webhook `POST /incoming-call` reste sur Fastify. Le TwiML pointe vers ce service si `VOICE_STREAM_URL` est defini (ex. `wss://host-public/media-stream`).

## V1 vs suite

V1 vise un appel **fonctionnel** (latence 2-3 s acceptable) :

- STT sur utterance complete (pas de partiels)
- LLM completion complete (pas de stream tokens)
- TTS phrase complete puis envoi par frames 20 ms
- Barge-in minimal : `clear` Twilio + cancel de la tache courante

Optimisations suivantes (interfaces deja separees) :

1. Streaming STT (partiels faster-whisper)
2. Streaming LLM (tokens -> phrases)
3. Streaming TTS (premiere audio plus tot)
4. Barge-in plus fin (coupure frame-accurate)
5. Montees en charge (workers GPU, file d'inference, plusieurs workers FastAPI)

## Lancer

Python 3.11+ recommande. Dependances runtime vocal :

```
cd voice-server
python -m venv .venv
source .venv/bin/activate
pip install -e ".[voice,dev]"
```

Systeme : `espeak-ng` (phonemizer Kokoro).

vLLM (exemple, l'id `--` doit etre celui que tu mettras dans VLLM_MODEL) :

```
vllm serve meta-llama/Llama-3.1-8B-Instruct --host 0.0.0.0 --port 8000
curl -s http://localhost:8000/v1/models
```

Copier l'id exact dans `VLLM_MODEL`. Un depot Llama gated exige un compte Hugging Face / `HF_TOKEN` au telechargement, pas pour l'inference locale.

Whisper : `cpu` + `int8` pour un premier test. Sur GPU : `WHISPER_DEVICE=cuda` et `WHISPER_COMPUTE_TYPE=float16`.

Comptes :

- Twilio : credentials appel
- Hugging Face : token eventuel pour telecharger certains modeles
- vLLM, Whisper, Silero, Kokoro, FastAPI, Fastify : locaux

Preflight (obligatoire avant un appel Twilio) :

```
voice-preflight
```

Teste FastAPI, Silero, Whisper, GET/POST vLLM `/v1`, Kokoro. Tant qu'une brique FAIL, ne pas appeler Twilio.

Voice Service :

```
VOICE_EAGER_LOAD=true voice-api
```

Health : `GET http://127.0.0.1:8090/health`

Au premier demarrage, Silero et Kokoro sont telecharges dans `~/.cache/mysmartfood-voice/` (Kokoro ~300 Mo).

## Limites V1

- Le prompt vocal du Voice Service est encore minimal (`app/prompts.py`). Le menu Pricing Fastify n'est pas injecte.
- La fin d'appel ne declenche plus l'extraction Fastify (`TranscriptionHandler`). A brancher ensuite via un POST metier si besoin.
- Fastify `/media-stream` proxy vers le Voice Service si `/health` est pret, sinon **FALLBACK GPT** OpenAI Realtime. Les handlers OpenAI restent en place pour ce fallback.
- Python 3.11 ou 3.12 recommande pour faster-whisper et Kokoro.

## Test d'un appel Twilio

Le webhook Fastify `/incoming-call` reste le TwiML. Le Stream Twilio doit viser Fastify (`wss://{PUBLIC_HOST}/media-stream`), pas le port Python : Fastify proxy vers le Voice Service, ou **FALLBACK GPT** OpenAI Realtime.

Simulation protocole Twilio (sans reseau) :

```
cd voice-server
python -m pytest tests/test_twilio_e2e.py tests/test_session_cleanup.py -q
```

