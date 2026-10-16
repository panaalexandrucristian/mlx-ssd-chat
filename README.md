# MLX SSD chat

Chat Qwen3 30B în terminal, cu streaming **exact** al experților de pe SSD,
cache dimensionat automat și monitor de memorie pentru macOS Apple Silicon.

```sh
ai --no-thinking
ai --mlx-memory-gb 8
ai --no-thinking --prompt 'What is 2 + 2? Reply briefly.'
```

Thinking este activ implicit. `/reset` resetează conversația; `q`, Ctrl+D și
Ctrl+C închid procesul. Contextul total este 4096 tokeni, inclusiv răspunsul.
Acesta este un chat text; nu execută tool-uri.

## Stare

- Sursele MLX-Flash 0.9.2 și patch-ul exact sunt versionate în acest repository.
- Build-ul local MLX-Flash, verificarea artefactelor și regresiile offline au trecut pe M3 / 16 GB / macOS 15.5.
- Build-ul nativ MLX necesită Xcode complet cu compilatorul Metal. Rezultatele
  efectiv obținute sunt consemnate în [`docs/validation.md`](docs/validation.md).
- Măsurătorile istorice din [`docs/reference-setup.md`](docs/reference-setup.md)
  provin de pe alt Mac (M4 Pro / 24 GB); nu sunt măsurători pe calculatorul curent.

## Instalare și build din surse

Necesită Apple Silicon, macOS ≥14, Git, `uv`, Python 3.12.13 și Xcode ≥15 cu
SDK compatibil cu sistemul. Instalează Xcode din App Store sau Apple Developer,
deschide-l pentru configurarea inițială și verifică:

```sh
xcrun --sdk macosx --find metal
xcrun --sdk macosx metal --version
```

Pentru o instalare Xcode nestandard, setează `DEVELOPER_DIR` înainte de build:

```sh
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
```

Instalează `uv` dacă lipsește, apoi:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
git clone https://github.com/panaalexandrucristian/mlx-ssd-chat.git
cd mlx-ssd-chat
uv python install 3.12.13
python3 scripts/build.py native --jobs 2
.venv/bin/python scripts/download_model.py
.venv/bin/python scripts/validate.py --real-model --label native
python3 scripts/install_launcher.py --profile native --shell
source ~/.zshrc
```

Build-ul folosește sursa MLX upstream nemodificată, configurație Release,
Metal activ, și produce separat wheel-urile `mlx`, `mlx-metal` și `mlx-flash`.
`upstream.lock.json` fixează commiturile și revizia modelului;
`requirements/runtime.txt` și `requirements/build.txt` fixează dependențele.
Sursa MLX se clonează în `.sources/mlx`, branch `local/exact-ssd`.
Sursa MLX-Flash se construiește din `vendor/mlx-flash`.

Artefactele, SHA-256, toolchain-ul, comenzile complete și logurile se păstrează
în directoare unice per build sub `artifacts/builds/`, cu loguri în `artifacts/logs/`.
Sursele MLX-Flash trebuie comise înainte de build; o copie curată numai a fișierelor
versionate este folosită ca intrare, cu hash pentru fiecare fișier.
`scripts/verify_runtime.py` compară fișierele instalate cu
wheel-urile locale, execută o operație GPU și verifică biblioteca MLX încărcată.
Build-ul refuză un checkout MLX modificat sau un compilator Metal absent.

Modelul ocupă aproximativ 17 GB pe SSD și se descarcă **explicit**. Chatul folosește
exclusiv revizia locală fixată; nu descarcă greutăți la pornire.

## Medii de referință și restaurare

Cele trei medii sunt independente:

| Profil | Mediu | MLX / Metal | MLX-Flash |
|---|---|---|---|
| `native` | `.venv` | Build local upstream | Build local cu patch |
| `flash` | `.venv-flash` | Wheel-uri oficiale fixate | Build local cu patch |
| `reference` | `.venv-reference` | Wheel-uri oficiale fixate | Wheel oficial + patch verificat |

```sh
python3 scripts/build.py reference
python3 scripts/build.py flash
.venv-flash/bin/python scripts/download_model.py
.venv-flash/bin/python scripts/validate.py --real-model --label flash
python3 scripts/install_launcher.py --profile flash --shell
```

Pentru revenire la profilul de referință:

```sh
python3 scripts/build.py reference
python3 scripts/install_launcher.py --profile reference --shell
```

Launcherul instalat indică explicit mediul ales. `bin/ai` din checkout folosește
implicit `.venv`; pentru alt mediu: `MLX_SSD_PYTHON="$PWD/.venv-flash/bin/python" bin/ai`.
Închide chatul înainte de rebuild sau schimbarea profilului.

## Memorie

Bugetele sunt în GB zecimali. Implicit: 2,8 GB MLX, 4 GB proces, 12 experți/strat.
Cu `--mlx-memory-gb 4`: 21 experți/strat și buget proces 5,2 GB.
Cu `--mlx-memory-gb 8`: 49 experți/strat și buget proces 9,2 GB.
`--process-memory-gb` poate suprascrie bugetul procesului.

Limita MLX este orientativă. Monitorul eșantionează RSS și physical footprint
la 50 ms și închide propriul proces cu codul 78 la depășire. Nu este un plafon
OS strict și nu limitează cache-ul de fișiere macOS. Capacitatea se calculează
din parametrii lazy înainte de încărcarea greutăților.

## Verificări și comparații

```sh
.venv/bin/python scripts/validate.py                 # fără modelul mare
.venv/bin/python scripts/validate.py --real-model   # chat, thinking, context, ieșire
uvx --from pyright==1.1.414 pyright --pythonpath .venv/bin/python \
  ai.py benchmark.py memory_guard.py cache_memory.py runtime.py export_setup.py scripts
.venv/bin/python scripts/compare_profiles.py --label native --repeats 2 --tokens 64
```

Comparațiile alternează 2,8 și 8 GB, cu proces 9,2 GB în ambele cazuri, același
prompt/seed/temperatură/limită de tokeni. Se măsoară separat cache fix 12 și
cache automat 12/49. Fiecare configurație are o încălzire exclusă; cache-ul de
fișiere OS nu este golit. Rapoartele JSON păstrează toate rulările, tokenii,
TTFT, viteza și memoria. Rulează aceeași comandă cu `.venv-reference/bin/python`
pentru comparația cu wheel-urile oficiale. Chatul și benchmark-ul folosesc
același lock, evitând rulările concurente ale modelului.

## Export autonom

```sh
python3 export_setup.py
python3 scripts/restore_export.py dist/mlx-ssd-setup.md /cale/noua/mlx-ssd-chat
```

Exportul include sursele urmărite de Git (inclusiv generatorul și restauratorul),
patch-urile, lock-urile, documentația și dovezile build/test disponibile.
Fișierele binare upstream sunt codificate base64. Restaurarea verifică inventarul
și hash-urile înainte de scriere. Wheel-urile se reconstruiesc, modelul se descarcă.
Pentru regenerarea exportului după modificări, adaugă fișierele noi în Git.
O restaurare fără Git poate regenera exportul folosind inventarul păstrat în
`EXPORT-MANIFEST.json`; fișierele deja inventariate sunt recitite de pe disc.

## Originea surselor

MLX: `36969c094354086e15d2a327a9c50feb20b9cf11` (`v0.32.4`).
MLX-Flash: `c3561916ae2e4fbfd2278483d8be570b4cce70f7` (`v0.9.2`).
Originalul `expert_streaming.py` din tag este identic byte-cu-byte cu wheel-ul
din document. Patch-ul, originalul și versiunea finală sunt în `reference/`.
Codul vendorizat păstrează licența MIT upstream în `vendor/mlx-flash/LICENSE`.
Detalii despre flux: [`docs/build.md`](docs/build.md).
