# Audit voice-server — assistant telephonique restaurant

Date: 2026-09-19
Machine d'audit: CPU uniquement. `nvidia-smi` absent. vLLM injoignable sur `localhost:8000`.
Aucun chiffre GPU, turbo, Parakeet, Nemotron, LLM TTFT ou appels/GPU n'est invente.

Objectif produit: conversation telephonique francaise naturelle.
Zero erreur n'existe pas. Le systeme minimise et mesure les erreurs.

---

## 1. Audit de l'architecture actuelle

```text
Appelant
  -> Twilio Media Streams (G.711 mu-law, 8 kHz, trames ~20 ms)
  -> Fastify webhook + proxy WebSocket /media-stream + streamToken
  -> FastAPI /media-stream
        |
        +-- VOICE_ORCHESTRATOR=native (BASELINE, reste le defaut apres A/B)
        |     CallSession
        |       codec mu-law <-> PCM16
        |       Silero VAD 8 kHz (voix, pas fin de tour)
        |       barge-in: cancel + Twilio clear
        |       resample 8k -> 16k
        |       preprocess optionnel (off par defaut)
        |       STT utterance (faster-whisper)
        |       turn heuristic (silence | heuristic | smart_turn pont)
        |       LLM stream phrase -> TTS stream -> mu-law 20 ms
        |       tools HTTP -> Fastify -> PostgreSQL/Mongo metier
        |
        +-- VOICE_ORCHESTRATOR=pipecat (CANDIDATE, pas passe en defaut)
              meme bootstrap que CallSession: streamToken, contexte, tools Fastify
              Twilio serializer + Smart Turn v3
              Whisper + vLLM OpenAI-compat + Kokoro
              greeting TTS + hangup report
              si pipecat-ai absent: fallback native
              A/B 2026-09-19: conversation live Twilio non mesuree; Smart Turn
              n'a pas tenu les phrases coupees sur le corpus Kokoro+mulaw
```

Cartographie detaillee:

| Couche | Implementation | Notes |
|---|---|---|
| Transport Twilio | KEEP | Media Streams mu-law 8 kHz. Fastify sonde puis proxifie. |
| WebSocket | KEEP | FastAPI accepte, Fastify peut envoyer du binaire UTF-8. |
| Codecs | KEEP | G.711 mu-law in/out, trames 20 ms. |
| Resampling | KEEP (renforce) | Anti-alias avant downsample 24k->8k. 8k->16k reste une interpolation. |
| Buffering | KEEP | Pre-roll ~200 ms-1 s. Phrase < 200 ms ignoree. |
| VAD | KEEP | Silero ONNX, 8 kHz natif, CPU. |
| End-of-turn | REPLACE (silence seul) | `TURN_DETECTOR=heuristic` par defaut. Smart Turn reserve a Pipecat. |
| STT | KEEP moteur / BENCHMARK modele | faster-whisper. Modele `small` mesure ici, inutilisable en CPU telephone. |
| LLM moteur | KEEP | vLLM OpenAI-compat + tool calls. Down pendant l'audit. |
| LLM modele | BENCHMARK | Identifiant configure, pas de TTFT mesure. |
| Tool calling | KEEP | Liste fermee -> dispatcher HTTP -> Fastify. Jamais SQL dans voice-server. |
| TTS | KEEP | Kokoro `ff_siwis`. Seul TTS FR self-host executable ici. |
| Interruption | KEEP (corrige) | VAD tourne pendant le TTS. `clear` Twilio + cancel tache. |
| Queues / locks | KEEP / risque | Verrous STT et TTS globaux par processus. |
| Timeouts | KEEP | Contexte restaurant 3 s. LLM HTTP 60 s. |
| Retry | KEEP | Fallback LLM OpenAI si configure. Pas de retry STT/TTS. |
| Cleanup | KEEP | `close` idempotente, `finally` WS, `close_all` au shutdown. |
| Metriques | KEEP (renforce) | `VoiceTelemetry` p50/p95 + evenements structures. |
| Logs | KEEP | `call_id`, `session_id`, `step`, `latency_ms`, `provider`, `model`. |

---

## 2. Problemes detectes

1. Barge-in mort: l'audio etait ignore pendant le TTS. Corrige (`_on_media` execute le VAD meme pendant la synthese).
2. Pipeline bloquant STT puis LLM entier puis TTS entier. Streaming phrase ajoute. Non mesure de bout en bout sur un vrai appel.
3. VAD confondu avec fin de tour (`VAD_SILENCE_MS=400`). Detecteur heuristique + timer `TURN_INCOMPLETE_WAIT_MS=800`.
4. Telemetrie sans percentiles. p50/p95 ajoutes.
5. Pas de corpus telephone reproductible. Ajoute (Kokoro -> mu-law 8 kHz -> bruit synthetique).
6. Verrous STT/TTS globaux: un seul utterance STT et un seul TTS a la fois par processus.
7. Whisper `small` CPU + audio telephone: WER ~1.0 et latence p50 27 s. Inutilisable pour le temps reel.
8. Bruit synthetique "street": 5/9 hypotheses vides + hallucinations Whisper (`Sous-titres realises par la communaute d'Amara.org`, `Sous-titrage ST' 501`).
9. vLLM et GPU L4 absents: aucun A/B turbo / Parakeet / Nemotron / LLM n'est tranche.
10. Pipecat 1.11.0 est installe (`pip install -e ".[pipecat]"`). Le defaut reste `native`.
11. A/B orchestrateur 2026-09-19 (fakes + Smart Turn corpus, pas d'appel Twilio reel):
    - Native: 0 false EOT immediat sur "Je voudrais reserver pour"; 5/5 flush a 801.3 ms (hold 800 ms); 5/5 combinaison avant hold; barge-in p50 0.01/0.04 ms; 1 trame media apres `clear`; false interrupt 5/5 sur burst 50 ms et bruit gaussien.
    - Smart Turn v3: 9/9 COMPLETE sur phrases full, cut 45 % et cut 30 %. Aucun INCOMPLETE mesure sur ce corpus. Latence modele p50 173.94 ms.
    - Pipecat `_report_hangup` : 2 hangups si appele deux fois. Pipeline live Twilio+Whisper: non mesure.
    - Tools: `check_availability` et `create_appointment` identiques; numero force; `update_appointment` / `cancel_appointment` absents des deux cotes.
12. `ARCHITECTURE.md` decrit encore partiellement l'etat pre-refresh (sections 5.5, 9, 18, 19).

---

## 3. Carte KEEP / REPLACE / BENCHMARK / REMOVE / UNKNOWN

| Composant | Verdict | Raison |
|---|---|---|
| Twilio Media Streams | KEEP | Transport production. Aucun benefice SIP mesure. Twilio accepte deja mu-law. |
| Fastify proxy /media-stream | KEEP | Sondes, fallback OpenAI Realtime, streamToken. |
| FastAPI CallSession | BASELINE, reste defaut | A/B 2026-09-19: turn-taking heuristique mesure (fakes). Pas de vainqueur live Twilio. |
| Pipecat | CANDIDATE, pas defaut | Parite bootstrap OK. Smart Turn toujours COMPLETE sur corpus coupe. Live Twilio non mesure. |
| G.711 mu-law 8 kHz | KEEP | Contrainte Twilio. |
| Resample lineaire 8k->16k | BENCHMARK | Suffisant pour charger Whisper. Qualite non isolee. |
| Preprocess AGC/gate | BENCHMARK | Default `off`. ON/OFF non re-execute (couteux CPU). |
| Silero VAD | KEEP | Leger, 8 kHz, CPU. Ce n'est pas un EOT. |
| Silence timeout = EOT | REPLACE | Remplace par `heuristic` + hold incomplet. |
| Heuristic turn + hold 800 ms | BASELINE / fallback | Mieux que VAD=EOT. Le hold fixe peut etre perceptible. |
| Smart Turn v3 | CANDIDATE | EOT audio Pipecat. A mesurer vs heuristic (false EOT, latence). |
| faster-whisper | KEEP (moteur) | Runtime local mature, compatible Pipecat. |
| Whisper small | REPLACE (cible GPU) | Baseline mesuree: WER 1.0219, p50 27096.6 ms, RTF 20.4 sur CPU. A remplacer apres A/B GPU. |
| Whisper large-v3 | BENCHMARK | Plus precis, trop lourd pour cohabiter ASR+LLM+TTS sur un L4 24 Go sans mesure. |
| Whisper large-v3-turbo | BENCHMARK | Candidat principal Whisper: 809M, decodeur 4 couches, ~8x plus rapide selon la model card OpenAI. Non telecharge ici. |
| NVIDIA Parakeet TDT 0.6B v3 | BENCHMARK | Model card 2025-08-14: FR FLEURS 5.15%, MLS 4.97%. NeMo absent. |
| NVIDIA Nemotron 3.5 ASR streaming 0.6B | BENCHMARK | Model card 2026-06-04: FR FLEURS ~9.03% a 1.12 s, chunks 80-1120 ms. Meilleur candidat streaming. Non installe. |
| NVIDIA Riva / NIM | BENCHMARK | Parakeet RNNT 1.1B multilingual, fr-FR, 8 kHz accepte. Licence/NIM + VRAM a mesurer. |
| vLLM | KEEP (moteur) | Serving OpenAI-compat et tool calls. |
| Modele LLM charge | BENCHMARK | vLLM down. Aucun TTFT. |
| Kokoro ff_siwis | KEEP | Apache-2.0, voix FR unique, TTFA proxy 3859 ms (inclut chargement). Pas d'alternative mesuree. |
| Piper / CosyVoice / XTTS | UNKNOWN | Non benchmarkes. Kokoro reste tant qu'un gain FR+TTFA n'est pas mesure. |
| ToolDispatcher -> Fastify | KEEP | Unique source de verite metier. |
| Audio maison parallele a Pipecat | REMOVE (cible) | Un seul orchestrateur actif par processus via `VOICE_ORCHESTRATOR`. Pas deux frameworks en parallele sur le meme appel. |

---

## 4. Technologies etudiees (sources officielles, 2025-2026)

| Technologie | Source | Conclusion utile |
|---|---|---|
| faster-whisper + Whisper small | Runtime local actuel | Moteur KEEP. Checkpoint `small` CPU refuse pour la prod telephonique. |
| Whisper large-v3-turbo | [openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo), [discussion #2363](https://github.com/openai/whisper/discussions/2363) | 809M, decodeur 4 couches, degradation mineure vs large-v3, pas de tache translate. Candidat A/B GPU n°1 cote Whisper. |
| Parakeet TDT 0.6B v3 | [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) | 25 langues EU, FR FLEURS 5.15%. Fort en batch, pas prouve en 8 kHz restaurant. |
| Nemotron 3.5 ASR streaming 0.6B | [nvidia/nemotron-3.5-asr-streaming-0.6b](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) (2026-06-04) | Streaming cache-aware, fr-FR transcription-ready, WER FLEURS ~9.03% a 1.12 s. Candidat streaming. |
| Riva / NIM Parakeet RNNT 1.1B | [docs.nvidia.com/nim/speech](https://docs.nvidia.com/nim/speech/26.02.0/asr/deploy-asr-models/parakeet-rnnt.html) | fr-FR, 8 kHz resample auto, streaming+offline. Cout VRAM/NIM inconnu ici. |
| Smart Turn v3 | [docs.pipecat.ai Smart Turn](https://docs.pipecat.ai/api-reference/server/utilities/turn-detection/smart-turn-overview), [pipecat-ai/smart-turn-v3](https://huggingface.co/pipecat-ai/smart-turn-v3) | EOT audio, pas transcript. CPU, ONNX, VAD `stop_secs=0.2`. Defaut Pipecat depuis 0.0.102. |
| Twilio serializer Pipecat | [TwilioFrameSerializer](https://docs.pipecat.ai/api-reference/server/services/serializers/twilio) | Compatible Media Streams. Pas de raison de quitter Twilio sans mesure. |
| Kokoro-82M | [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M), [VOICES.md](https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md) | Apache-2.0. FR: une voix `ff_siwis`, <11 h, note B-. espeak-ng GPL pour le G2P. |
| Silero VAD | Runtime ONNX actuel | Conserve pour la detection de voix. |

Les articles "Top 10 STT" n'ont pas ete utilises pour decider.

---

## 5. Benchmarks realises

Commandes:

```bash
cd voice-server
python -m benchmarks.generate_corpus --conditions clean,street
python -m benchmarks.benchmark_stt --provider faster-whisper --preprocess off
python -m benchmarks.benchmark_latency
```

Corpus: 9 phrases restaurant FR, 2 conditions, 18 WAV PCM16 8 kHz apres aller-retour mu-law.
Fichiers: `benchmarks/expected_transcripts.json`, `benchmarks/audio/`, `benchmarks/results/`.

Limite methodologique: parole synthetique Kokoro, pas voix humaine; bruit "street" = gaussien, pas rue reelle. Un WER studio 48 kHz n'a pas ete utilise.

Non executes (dependances ou GPU absents):

- `whisper-large-v3-turbo`
- `parakeet`
- Nemotron streaming
- preprocess `agc` / `gate`
- vLLM TTFT
- Pipecat Smart Turn sur appel Twilio (mesure hors pipeline: corpus Kokoro, pas humain)
- concurrence multi-appels

---

## 6. Resultats mesures

Source: `benchmarks/results/stt_faster-whisper_off.json` et `latency_local.json`.

### STT faster-whisper `small` / preprocess `off` / CPU int8

| Ensemble | Samples | WER moyen | Latence p50 | Latence p95 | RTF moyen | Hypotheses vides |
|---|---:|---:|---:|---:|---:|---:|
| All (clean+street) | 18 | 1.0219 | 27096.6 ms | 79138.4 ms | 20.404 | 5 |
| clean | 9 | 0.9604 | 5450.8 ms | — | — | 0 |
| street | 9 | 1.1944 | 48503.3 ms | — | — | 5 |

Exemples clean:

| Id | Reference | Hypothese | WER |
|---|---|---|---:|
| table_quatre_2030 | Bonsoir, vous avez une table pour quatre vers vingt heures trente ? | Le chien, vous avez une table pour quatre veilleres trente. | 0.4545 |
| correction_six | Finalement on sera six. | Vous n'en avez pas le sens. | 1.5 |
| nom_nguyen | C'est pour Monsieur Nguyen. | C'est pour nos filles et bien. | 1.0 |
| annuler_soir | Vous pouvez annuler la reservation de ce soir ? | Vous pouvez lancer la video, la reservation de ce soir. | 0.375 |

Street: plusieurs silences et hallucinations de sous-titrage.

### Latence locale CPU

| Metrique | Valeur | Perimetre |
|---|---:|---|
| VAD load + infer | 107 ms | Silero ONNX, 20 fenetres silence |
| TTS TTFA proxy | 3859 ms | Kokoro "Oui. Pour combien de personnes ?" inclut le chargement |
| TTS samples | 11435 | PCM 8 kHz |
| LLM TTFT | non mesure | vLLM down |
| TTS TTFA streaming hors load | non mesure | |
| End-turn -> first audio | non mesure | pas d'appel Twilio instrumente |
| VRAM | non mesure | pas de GPU |
| GPU util | non mesure | |
| Calls / GPU | non mesure | |

Tests conversationnels (fakes, pas modeles reels): `tests/test_conversation_evals.py`, scenarios `benchmarks/scenarios/*.json`.

---

## 7. Tableau de comparaison (mesures seulement)

| Metric | Current (mesure) | Candidate | Winner |
|---|---|---|---|
| French accuracy | WER clean 0.9604 sur corpus Kokoro+mulaw | turbo / Parakeet / Nemotron: non mesures | non tranche |
| Telephone accuracy | WER global 1.0219 | non mesure | non tranche |
| WER | 1.0219 | — | current uniquement mesure, et il est mauvais |
| STT latency p50 | 27096.6 ms CPU | — | current inacceptable pour le realtime |
| STT latency p95 | 79138.4 ms CPU | — | current inacceptable |
| LLM TTFT | non mesure | non mesure | — |
| TTS TTFA | 3859 ms proxy CPU (load inclus) | non mesure | Kokoro seul mesure |
| End-turn -> audio | non mesure | non mesure | — |
| VRAM | non mesure | non mesure | — |
| Concurrent calls | non mesure | non mesure | verrous STT/TTS = 1 utterance / processus |

Conclusion produit: on ne remplace pas Whisper small par turbo ou Parakeet dans le defaut, faute de mesure GPU. On constate que small CPU est inapte. Le prochain A/B obligatoire sur L4 est:

```text
faster-whisper small (cuda/float16)
vs faster-whisper large-v3-turbo
vs parakeet-tdt-0.6b-v3
vs nemotron-3.5-asr-streaming-0.6b
```

sur le meme corpus mu-law 8 kHz, puis sur 10 vrais extraits d'appels.

---

## 8. Raisons techniques de chaque choix

1. **Twilio conserve.** Changer de SIP sans mesurer la latence, le jitter et le barge-in n'ameliore pas la comprehension. Twilio fournit deja mu-law 8 kHz, le vrai goulot est ASR/EOT/LLM, pas le transport.
2. **Fastify reste la seule source metier.** Invariant de securite: le LLM ne touche pas PostgreSQL.
3. **CallSession = BASELINE, reste le defaut.** Pipecat = CANDIDATE, pas passe en defaut. Les deux partagent `build_call_plan`. L'A/B 2026-09-19 n'a pas montre un gain conversationnel clair de Pipecat (Smart Turn toujours COMPLETE sur audio coupe; barge-in/cleanup live non mesures). `VOICE_ORCHESTRATOR=native` inchange.
4. **Silero conserve pour la voix, pas pour le tour.** Distinguer VAD et EOT est une exigence produit. L'heuristique evite de repondre sur "Je voudrais reserver pour...".
5. **faster-whisper conserve comme moteur.** Il sert small et turbo sans reecrire le pipeline. Le checkpoint `small` est la baseline, pas la cible.
6. **Pas de switch ASR par defaut.** Parakeet (5.15% FLEURS FR) et Nemotron streaming (9.03% FLEURS FR, mais chunks 80 ms) sont officiellement pertinents. Les WER fournisseurs ne sont pas du telephone restaurant 8 kHz. Mesure locale d'abord.
7. **vLLM conserve comme moteur.** Le modele charge n'est pas valide: vLLM etait down.
8. **Kokoro conserve.** Seul TTS FR self-host licence Apache execute ici. Une seule voix FR, note B-. Remplacement seulement si un candidat bat TTFA et la prosodie des heures/noms.
9. **Streaming phrase, pas mot a mot.** `TTS_CHUNKING=sentence` pour garder la prosodie.
10. **Preprocess off.** Un denoisage agressif casse souvent Whisper. ON/OFF a mesurer sur GPU, pas a activer par gout.

---

## 9. Nouvelle architecture

```text
.env
  STT_PROVIDER=faster-whisper|whisper-large-v3-turbo|parakeet
  LLM_PROVIDER=vllm|openai|openai-compat
  TTS_PROVIDER=kokoro
  TURN_DETECTOR=silence|heuristic|smart_turn
  VOICE_ORCHESTRATOR=native|pipecat
  AUDIO_PREPROCESS=off|agc|gate

voice-server/app
  providers.py          factory unique
  session/call.py       orchestrateur native
  pipeline/pipecat_runner.py  orchestrateur optionnel
  stt/faster_whisper.py + stt/parakeet.py
  llm/vllm.py           stream + tools
  tts/kokoro.py         synthesize_stream
  turn/detector.py      VAD != EOT
  tools/dispatcher.py   HTTP Fastify seulement
  observability.py      evenements structures
```

Un appel = un orchestrateur. Le metier reste:

```text
LLM -> tool Pipecat ou CallSession -> ToolDispatcher HTTP -> Fastify -> base
```

Cible L4 24 Go (a mesurer, pas imposee):

```text
1 GPU
  ASR (turbo ou Parakeet/Nemotron, un seul charge)
  vLLM (modele court, tool calling)
  TTS Kokoro de preference CPU/ONNX pour liberer la VRAM
```

Ne pas charger small + turbo + Parakeet + un grand LLM sur le meme GPU.

---

## 10. Fichiers modifies

- `app/main.py` — factory moteurs, branche `/media-stream` native|pipecat
- `app/config.py` — providers, streaming, turn, barge-in
- `app/session/bootstrap.py` — streamToken, contexte, tools, numero (partage native/Pipecat)
- `app/session/call.py` — barge-in, heuristic hold, stream LLM/TTS, events
- `app/providers.py` — selection STT/TTS/LLM/turn/orchestrateur
- `app/llm/vllm.py`, `app/llm/fallback.py` — `complete_stream`
- `app/tts/kokoro.py` — `synthesize_stream`
- `app/stt/faster_whisper.py` — `initial_prompt`
- `app/prompts.py` — style telephone
- `app/audio/resample.py` — anti-alias downsample
- `app/telemetry.py` — p50/p95
- `app/runtime.py` — turn + noms de providers
- `app/pipeline/pipecat_runner.py` — Smart Turn + tools Fastify
- `.env.example`, `pyproject.toml`
- tests: barge-in, providers, chunking, turn, streaming, evals, telemetry, pipecat

---

## 11. Fichiers ajoutes

- `app/audio/chunking.py`
- `app/audio/preprocess.py`
- `app/turn/detector.py`
- `app/observability.py`
- `app/conversation/corrections.py`
- `app/stt/parakeet.py`
- `app/pipeline/pipecat_runner.py`
- `app/ab/orchestrator.py`, `app/ab/doubles.py`
- `benchmarks/ab_orchestrator.py`
- `benchmarks/wer.py`, `generate_corpus.py`, `benchmark_stt.py`, `benchmark_latency.py`
- `benchmarks/expected_transcripts.json`
- `benchmarks/scenarios/{reservation,interruption,correction}.json`
- `benchmarks/results/stt_faster-whisper_off.json`
- `benchmarks/results/latency_local.json`
- `docs/VOICE_AUDIT.md`

---

## 12. Variables d'environnement

Nouvelles / critiques:

```text
STT_PROVIDER=faster-whisper
TTS_PROVIDER=kokoro
VAD_PROVIDER=silero
TURN_DETECTOR=heuristic
VOICE_ORCHESTRATOR=native
AUDIO_PREPROCESS=off
PARAKEET_MODEL=nvidia/parakeet-tdt-0.6b-v3
LLM_STREAM=true
TTS_STREAM=true
TTS_CHUNKING=sentence
TURN_INCOMPLETE_WAIT_MS=800
BARGE_IN_ENABLED=true
WHISPER_MODEL_SIZE=small
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8
WHISPER_LANGUAGE=fr
LLM_PROVIDER=vllm
VLLM_BASE_URL=http://localhost:8000/v1
VLLM_MODEL=<id exact GET /v1/models>
```

Sur GPU L4, viser ensuite (apres A/B, pas maintenant):

```text
WHISPER_DEVICE=cuda
WHISPER_COMPUTE_TYPE=float16
STT_PROVIDER=whisper-large-v3-turbo   # seulement si le benchmark gagne
```

---

## 13. Commandes de demarrage

```bash
cd voice-server
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[voice,dev]"
# optionnel: python -m pip install -e ".[pipecat]"
sudo apt-get install -y espeak-ng
cp .env.example .env   # renseigner NODE_API_KEY, TWILIO_AUTH_TOKEN, VLLM_*

voice-preflight
voice-api
# equivalent: python -m app.main

curl http://127.0.0.1:8090/live
curl -i http://127.0.0.1:8090/ready
curl http://127.0.0.1:8090/health
```

vLLM separe:

```bash
vllm serve <modele> --host 127.0.0.1 --port 8000 \
  --enable-auto-tool-choice --tool-call-parser <parser>
curl http://127.0.0.1:8000/v1/models
```

Tests:

```bash
python -m pytest -q
python -m benchmarks.generate_corpus
python -m benchmarks.benchmark_stt --provider faster-whisper --preprocess off
python -m benchmarks.benchmark_latency
```

---

## 14. Procedure de test Twilio

1. Fastify + voice-server + vLLM healthy (`/ready` 200, `engines.ready` true).
2. Webhook Twilio vers Fastify `/incoming-call` (ou gateway slug).
3. TwiML `<Connect><Stream>` vers le proxy Fastify, pas directement le port 8090 public.
4. Verifier `customParameters.instanceId` + `streamToken`.
5. Scenario reservation: "Je voudrais reserver pour quatre demain a vingt heures."
6. Verifier un `tool:check_availability` ou `create_appointment` selon le workflow, jamais une confirmation sans succes d'outil.
7. Correction: "Nous serons quatre." puis "Ah non, six." -> `nombrePersonnes=6`.
8. Barge-in: parler pendant l'accueil. Attendu: log `interruption_detected`, `clear` Twilio, `audio_buffer_cleared`, nouvelle ecoute.
9. Phrase incomplete: "Je voudrais reserver pour..." + pause < 800 ms + "six personnes." Le bot ne doit pas repondre au milieu.
10. Numero masque: le bot demande le telephone.
11. Couper vLLM: fallback LLM Python ou fallback appel Fastify Realtime selon la config.
12. Controler que logs/monitoring n'embarquent pas de numero ni de transcript complet.

---

## 15. Performances mesurees

Voir section 6. Resume: ASR CPU `small` hors contrat realtime (p50 27 s, WER ~1.0). VAD 107 ms. Kokoro TTFA proxy 3.86 s dont load. Reste non mesure.

---

## 16. Limites restantes

- Pas de 0% d'erreur. Noms (Nguyen, Schmitt), heures et bruit restent fragiles.
- Corpus synthetique: sous-estime la parole humaine, surestime le bruit gaussien.
- Pas d'A/B GPU. Le defaut `small` n'est pas la cible production.
- Pas de STT partiel/streaming en native (utterance complete).
- Verrous STT/TTS: concurrence = scale-out de replicas, pas un seul process.
- Smart Turn mesure seulement sur corpus Kokoro+mulaw (toujours COMPLETE, y compris cut 30/45 %). Pas de voix humaine telephone.
- LLM TTFT, tool-call error rate, hallucinations: non mesures (vLLM down).
- Preprocess ON/OFF non mesure.
- Confirmation metier toujours guidee par le prompt, pas un automate serveur.
- Signatures Twilio et auth de certains endpoints internes: dette pre-existante (`ARCHITECTURE.md` section 22).

---

## 17. Prochaines optimisations (ordre)

1. Fait: A/B Native vs Pipecat (fakes + Smart Turn corpus). Defaut inchange.
2. GPU L4: small CUDA vs large-v3-turbo vs Parakeet vs Nemotron streaming.
3. Mesurer `partial_first_ms` (premier partiel) en plus du WER final.
4. Vrais extraits humains mu-law 8 kHz, y compris phrases inachevees pour Smart Turn.
5. Concurrence 1/2/4/8 appels, p95. Les verrous STT/TTS globaux sont un risque structurel.
6. Revoir Smart Turn seulement sur parole humaine. Le hold 800 ms n'est pas une constante optimale.
7. Si un futur A/B live Twilio montre un gain conversationnel stable de Pipecat, alors seulement proposer le defaut.

Regle finale respectee: Whisper, vLLM et Kokoro n'ont pas ete "reussis a integrer". Ils restent la baseline executable. Les mesures disent que Whisper small CPU ne doit pas etre la cible. Le remplacement attend un A/B GPU sur audio telephone.

---

## 18. A/B orchestrateur Native vs Pipecat (2026-09-19)

Source: `benchmarks/results/ab_orchestrator.json`.
Commande: `python -m benchmarks.ab_orchestrator`.
Perimetre: memes tools Fastify, meme streamToken, meme contexte restaurant, memes fakes STT/LLM/TTS cote Native, Smart Turn v3 sur le corpus telephone existant.
Critere interdit: `speech_end -> first_audio` avec Whisper small CPU (~27 s) n'est pas utilise pour departager.

`VOICE_ORCHESTRATOR=native` n'a pas ete change.

### Tableau (valeurs mesurees uniquement)

| Metric | Native | Pipecat |
|---|---:|---:|
| False EOT | 0/5 immediat; 5/5 flush apres hold 800 ms | Smart Turn: 9/9 COMPLETE sur cut 45 % et cut 30 % (proxy inacheve). Live Twilio: non mesure |
| Missed EOT | 0/5 sur "Oui." et clause combinee avant hold | 0/9 incomplete sur phrases clean completes. Live Twilio: non mesure |
| EOT decision latency | p50 0.42 ms (heuristic + FakeSTT) | p50 173.94 ms (ONNX Smart Turn, corpus) |
| Barge-in detection | p50 0.01 ms (5/5, FakeVAD) | non mesure |
| Buffer clear latency | p50 0.04 ms | non mesure |
| False interruptions | silence 0/5; bas niveau 0/5; burst 50 ms 5/5; gaussien std400 5/5 | non mesure (souffle / resto / echo: non mesure des deux cotes) |
| Cleanup failures | 0/5; hangup x1; 0 tache restante | reject token WS 1008 OK; hangup x2 si `_report_hangup` double; pipeline live: non mesure |
| Tool errors | check+create OK; numero force; update/cancel absents | identique (memes tools Fastify) |
| Orchestration overhead | speech_end -> llm p50 0.62 ms (tour complet); 801.3 ms (incomplet = hold) | infer Smart Turn p50 173.94 ms. speech_end -> first_audio live: non mesure, non discriminant |
| Code realtime proprietaire | 944 LOC (call+turn+buffer+protocol) | 254 LOC glue + pipecat-ai 1.11.0 |

### Turn-taking

Native, 5 repeats, FakeSTT:

- `"Je voudrais reserver pour"` : pas de reponse immediate (0/5). Reponse apres 801.3 ms (5/5).
- Pause courte puis `"six personnes."` avant 800 ms : 5/5 tours combines, 0 premature.
- Pause longue > 800 ms puis deuxieme clause : 5/5, le bot a deja repondu.
- `"euh"` puis suite : 5/5 hold puis combine.
- `"Attendez"` puis correction : 5/5 hold puis combine.
- Utterance < 200 ms : 5/5 droppee.

Pipecat Smart Turn v3 sur 9 WAV clean + 9 street (Kokoro -> mu-law, pas voix humaine):

- full clean: 9 COMPLETE / 0 INCOMPLETE
- cut 45 %: 9 COMPLETE / 0 INCOMPLETE
- cut 30 %: 9 COMPLETE (hesitation_cut30_incomplete = 0)
- street full: 9 COMPLETE / 0 INCOMPLETE

Le proxy "phrase inachevee" n'a donc **pas** ete reconnu comme INCOMPLETE. On ne peut pas en conclure que Smart Turn est mauvais sur de la parole humaine. On peut conclure qu'il n'a pas gagne le critere false EOT sur le corpus disponible.

### Barge-in (Native seulement)

Scenario bot en train de parler, VAD `speech_started`:

- `speech_start -> interruption_detected` p50 0.01 ms
- `speech_start -> audio_buffer_cleared` p50 0.04 ms
- `clear` Twilio present 5/5
- 1 trame media encore envoyee apres le dernier `clear` (p50)
- `_assistant_speaking` faux apres coup 5/5

Pipecat: `allow_interruptions=True` dans le glue, **aucun timing live**.

### False interruptions

Native, QuietVad seuil 80, 5 repeats:

- silence: 0
- amplitude 40: 0
- burst 50 ms amplitude 12000: 5 (aucune duree minimale cote barge-in)
- bruit gaussien std 400: 5

Souffle humain, bruit restaurant reel, echo du bot: **non mesures**.

### Cleanup

Native `close()` deux fois: 0 echec, hangup Fastify une seule fois, 0 tache restante.

Pipecat: token invalide ferme le WS 1008 sans pipeline. `_report_hangup` n'est pas idempotent (2 envois mesures). Fermeture live (WS, tasks, queues): **non mesuree** (Whisper small n'a pas ete charge).

### Tools

Les deux orchestrateurs passent par `ToolDispatcher` Fastify:

- `check_availability`: OK
- `create_appointment`: OK, `telephone` force a `0612345678`
- `update_appointment`: `Fonction inconnue` des deux cotes
- `cancel_appointment`: `Fonction inconnue` des deux cotes

Le resultat metier est identique. Le LLM ne peut pas remplacer le numero session.

### Complexite (factuel)

Native: 944 LOC realtime maison (session, heuristique EOT, hold, barge-in, locks, hangup idempotent). Toute la maintenance turn-taking est locale.

Pipecat: 254 LOC de glue. Utilise Pipeline, Twilio serializer, Smart Turn v3, Silero VAD, Whisper/vLLM/Kokoro services, `allow_interruptions`. Adaptations specifiques: `build_call_plan`, tools Fastify, numero force, greeting, hangup. Dependance framework 1.11.0. Hangup non idempotent.

### Verdict

Pipecat ne gagne pas clairement la conversation temps reel sur les mesures disponibles. Native reste le defaut.

Raisons:

1. Smart Turn n'a tenu aucune phrase coupee sur le corpus actuel (0 INCOMPLETE / 18 coupes).
2. Barge-in, leftover audio, false interrupt et cleanup live Pipecat sont non mesures. On ne peut pas declarer une victoire de stabilite.
3. Native gere le scenario "pour ... six personnes" avant 800 ms (5/5) et le hangup une fois (5/5).
4. Native a un risque mesure: interruption sur burst court et bruit. Ce n'est pas un argument pour Pipecat tant que le meme test n'existe pas cote framework.
5. Moins de glue Pipecat (254 vs 944) ne suffit pas: ce n'est pas un critere de conversation.
6. `speech_end -> first_audio` n'a pas ete utilise: Whisper small CPU dominerait les deux.

Prochaine phase separee, pas une refonte d'orchestrateur: A/B ASR sur L4 (small CUDA vs turbo vs Nemotron streaming vs Parakeet).

---

## 19. Corrections post-A/B (2026-09-19)

Native n'est pas declare architecturalement superieur. Il est le seul chemin assez valide pour rester le defaut.

### Barge-in confirme

`BARGE_IN_CONFIRM_MS=160` (bande 120-200, **pas un optimum**). Sweep: `benchmarks/results/barge_in_confirm.json`.

| confirm_ms | burst 50 ms | parole 200 ms | bruit soutenu 250 ms |
|---:|:---:|:---:|:---:|
| 50 | interrupt | interrupt | interrupt |
| 80-200 | hold | interrupt | interrupt |
| 250 | hold | hold | interrupt |

Mesure session apres correctif (5 repeats):

- burst 50 ms: 0/5 interrupt (etait 5/5)
- silence / bas niveau: 0/5
- gaussien 250 ms: 5/5 encore (VAD le classe parole continue)
- barge-in parole 200 ms+: detection p50 0.04 ms, clear p50 0.07 ms

Le bruit restaurant soutenu n'est pas resolu par la duree seule.

### Epoch de generation

`PlaybackEpoch` + verif apres chaque `await` avant `send`. Frames media apres dernier `clear`: p50 **0** (etait 1).

### Hangup Pipecat

`_report_hangup` idempotent: 1 envoi sur double appel (etait 2).

### Smart Turn : chemin verifie, verdict corpus non retenu

Pipecat 1.11.0, chemin reel:

```text
Twilio mu-law 8 kHz
  -> TwilioFrameSerializer.ulaw_to_pcm
  -> PCM 16 kHz (PipelineParams audio_in=16000)
  -> InputAudioRawFrame
  -> TurnAnalyzerUserTurnStopStrategy.append_audio(frame, vad_user_speaking)
  -> analyze_end_of_turn au VAD stop (stop_secs=0.2)
```

- `set_sample_rate(audio_in_sample_rate)` = 16 kHz.
- Smart Turn recoit **tout le tour courant**, pas seulement le dernier segment VAD.
- Silero `stop_secs` defaut et config explicite: **0.2**.
- SOXR stream: une trame 20 ms peut sortir vide; 200 ms de mu-law produit du PCM 16 kHz (mesure: 1600 samples 8 kHz -> 1516 samples 16 kHz).

Le cut Kokoro 30/45 % de l'A/B n'est donc pas equivalent a un humain qui hesite. Il ne sert plus a choisir l'orchestrateur.

### L4

`nvidia-smi` absent sur cette machine. Aucun WER/latence GPU invente.
Probe: `benchmarks/results/stt_l4_probe.json`.
L'A/B STT GPU n'est plus la cible. Voir section 19.

`VOICE_ORCHESTRATOR=native` inchange jusqu'au GO Omni.

---

## 19. Pivot Qwen2.5-Omni INT4 S2S (2026-09-19)

Cible produit: Twilio -> Qwen2.5-Omni-7B GPTQ-Int4 -> Twilio. Plus de Whisper, LLM texte, Kokoro, Smart Turn ni second orchestrateur sur le chemin principal.

Audit papier (cartes officielles, pas une mesure L4):

| Sujet | Verdict | Preuve |
|---|---|---|
| VRAM INT4 15 s video | 11.64 Go, L4 24 Go OK | [GPTQ-Int4](https://huggingface.co/Qwen/Qwen2.5-Omni-7B-GPTQ-Int4) |
| VRAM INT4 30 s video | 17.43 Go, L4 OK avec marge | meme carte |
| VRAM INT4 60 s video | 29.51 Go, L4 KO | meme carte |
| BF16 7B | 31.11 Go / 15 s, L4 KO | README Qwen2.5-Omni |
| Streaming local `generate()` | waveform 24 kHz complet en fin de tour | Transformers + demo officielle |
| token2wav INT4 | stream interne anti-prealloc, pas d'API Twilio 20 ms | carte GPTQ |
| vLLM actuel `v0.10.2` | thinker texte seulement, pas de talker | README Omni + compose Scaleway |
| Function calling local | pas d'argument `tools` sur `generate()` | docs Transformers / README |
| FR ASR | Common Voice 15 FR WER 7.5 (officiel 7B) | README Omni |
| FR TTS | Seed-TTS zh/en seulement. Chelsie / Ethan. Inconnu au telephone FR | README Omni |

L4 production au moment de l'audit: `https://voice.mysmartfood.fr/health` repond `engines.ready=true`, `llm_provider_active=openai` (`gpt-4o-mini`). L'ancien pipeline est vivant. Omni INT4 ne peut pas cohabiter. Aucun poids Omni n'a ete charge.

Prototype lab, hors Twilio / Fastify: `prototypes/omni_s2s/proto_s2s.py`.
Probe: `benchmarks/results/omni_s2s_probe.json` (`status=not_measured` sur cette machine, pas de CUDA).

Integration Twilio interdite tant que le proto n'a pas mesure VRAM, latence S2S, francais parle, et tool calls thinker.
