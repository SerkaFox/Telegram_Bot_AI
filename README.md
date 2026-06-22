# tg_comfy_bot

Telegram-бот, который ставит задачи в очередь к ComfyUI и присылает результат обратно в чат.
Бот сам по себе не содержит весов моделей — он только собирает/патчит JSON-воркфлоу и
дёргает ComfyUI REST API (`/prompt`, `/history`, `/upload/image`). Все модели должны лежать
в обычных папках ComfyUI (`models/checkpoints`, `models/loras`, `models/vae`, и т.д.).

Этот файл — шпаргалка на случай переустановки сервера: что за нода/модель нужна для
какого режима бота и где её взять заново.

## Режимы бота → воркфлоу

| Режим бота      | Файл воркфлоу              | Движок                          |
|-----------------|-----------------------------|----------------------------------|
| `video`         | `workflow_video.json`       | WAN 2.2 I2V (GGUF)               |
| `ltx_sulphur`   | `LTX2.3_2.json`              | LTX-2.3 Sulphur (видео+аудио)    |
| `ltx_eros`      | `workflow_ltx_eros.json`     | LTX-2.3 10Eros (видео+аудио)     |
| `image` (Edit Photo) | собирается в коде (`build_image_edit_workflow`) | Qwen-Image-Edit-2509 |
| `mopmix`        | `workflow_mopmix.json`       | SDXL bigASP 2.5 (img2img)        |
| `mopmix_duo`    | `workflow_mopmix_duo.json`   | SDXL bigASP 2.5 + ReActor        |

`tools/ui_to_api.py` — общий конвертер ComfyUI UI-формата (с подграфами, Set/Get-нодами,
выключенными нодами) в плоский API-формат. Использовался для пересборки `workflow_ltx_eros.json`
из `LTX2.3NotApi.json` и `workflow_mopmix*.json` из `mopMixWorkflows_BigASP25.json`. Если
воркфлоу снова придётся пересобирать из экспорта ComfyUI — это тот инструмент:

```
python3 tools/ui_to_api.py исходник_UI.json результат_API.json
```
(нужен запущенный ComfyUI на `http://127.0.0.1:8188` — конвертер дёргает `/object_info`).

## Восстановление: custom nodes

Клонировать в `ComfyUI/custom_nodes/`:

```
git clone https://github.com/rgthree/rgthree-comfy
git clone https://github.com/pythongosssss/ComfyUI-Custom-Scripts comfyui-custom-scripts
git clone https://github.com/yolain/ComfyUI-Easy-Use comfyui-easy-use
git clone https://github.com/cubiq/ComfyUI_essentials comfyui_essentials
git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite comfyui-videohelpersuite
git clone https://github.com/city96/ComfyUI-GGUF
git clone https://github.com/stduhpf/ComfyUI-WanMoeKSampler
git clone https://github.com/kijai/ComfyUI-KJNodes
git clone https://github.com/Lightricks/ComfyUI-LTXVideo
git clone https://github.com/GACLove/ComfyUI-VFI
git clone https://github.com/Comfy-Org/Nvidia_RTX_Nodes_ComfyUI
git clone https://github.com/huchukato/ComfyUI-QwenVL-Mod
git clone https://github.com/danTheMonk/comfyui-int-and-float
git clone https://github.com/Smirnov75/ComfyUI-mxToolkit comfyui-mxtoolkit
git clone https://github.com/LAOGOU-666/Comfyui-Memory_Cleanup comfyui_memory_cleanup
git clone https://github.com/gseth/ControlAltAI-Nodes controlaltai-nodes
git clone https://github.com/TenStrip/10S-Comfy-nodes.git
git clone https://github.com/evanspearman/ComfyMath
git clone https://github.com/kijai/ComfyUI-PromptRelay.git
git clone https://github.com/whatdreamscost/whatdreamscost-comfyui.git
git clone https://github.com/ltdrdata/ComfyUI-Impact-Pack comfyui-impact-pack
git clone https://github.com/ltdrdata/ComfyUI-Impact-Subpack comfyui-impact-subpack
git clone https://github.com/Gourieff/ComfyUI-ReActor.git
git clone https://github.com/vrgamegirl19/comfyui-vrgamedevgirl
git clone https://github.com/kijai/ComfyUI-Florence2 comfyui-florence2
git clone https://github.com/lrzjason/Comfyui-QwenEditUtils qweneditutils
```

После клонирования у каждого пакета поставить зависимости (`pip install -r requirements.txt`
в venv ComfyUI) и перезапустить `comfyui.service`.

`whatdreamscost-comfyui` (содержит `LTXSequencer`, `MultiImageLoader`) и
`ComfyUI-PromptRelay` / `10S-Comfy-nodes` — специфичны для `ltx_eros`, без них этот режим
не запустится вообще (нода `class_type` не найдётся).

## Восстановление: модели по режимам

### `video` (WAN 2.2 I2V)
- `wan22EnhancedNSFWSVICamera_nsfwFASTMOVEV2Q4KMH.gguf` / `...Q4KML.gguf` — UNET (high/low noise), CivitAI, точная ссылка не сохранена — искать по имени файла.
- `nsfw_wan_umt5-xxl_fp8_scaled.safetensors` — текстовый энкодер (CLIP), источник не сохранён.
- `wan_2.1_vae.safetensors` — VAE, источник не сохранён.
- Опциональные LoRA (кнопка LoRA в боте, `VIDEO_LORA_OPTIONS` в коде) — список из ~15 NSFW LoRA для WAN 2.2, все предустановлены до этой сессии, точные ссылки не сохранены. Имена файлов — в `telegram_comfyui_bot.py`, искать по имени на CivitAI.

### `ltx_sulphur` (LTX2.3_2.json)
- `sulphur_distil_bf16.safetensors` — чекпоинт/текстовый энкодер/audio VAE (один файл, грузится тремя разными лоадерами). Источник: **https://huggingface.co/SulphurAI/Sulphur-2-base**
- `DR34ML4Y_LTXXX_V2.safetensors` — LoRA, источник не сохранён (искать по имени).
- `LTX2.3_reasoning_I2V_V3.safetensors`, `LTX2.3_Multi_step_video_reasoning_V0.1.safetensors` — LoRA, ссылки дал пользователь напрямую в чате, не сохранены отдельно — искать по имени файла.
- `LTX2.3-NSFWMOTION_00750.safetensors` — LoRA, **на сервере отсутствует** (в логах ComfyUI: `Lora "..." not found, skipping`), но указан в воркфлоу — не блокирует генерацию, просто не применяется.
- `ltx-2.3-spatial-upscaler-x2-1.1.safetensors` — латентный апскейлер. Должен лежать в `models/latent_upscale_models/` (НЕ в `upscale_models/` — это типичная ошибка, `LatentUpscaleModelLoader` не найдёт его не в той папке).
- `taeltx2_3.safetensors` — TAE VAE превью, источник не сохранён.
- `flownet.pkl` — модель RIFE-интерполяции (часть `ComfyUI-VFI`), обычно скачивается автоматически при первом запуске ноды.

### `ltx_eros` (workflow_ltx_eros.json)
- `10Eros_v1.2_fp8mixed_learned.safetensors` (чекпоинт) — Источник: **https://huggingface.co/TenStrip/LTX2.3-10Eros**
- `ltx-2.3-22b-distilled-lora-1.1_fro90_ceil72_condsafe.safetensors` (LoRA) — Источник: **https://huggingface.co/TenStrip/LTX2.3_Distilled_Lora_1.1_Experiments**
- `gemma_3_12B_it_fp8_e4m3fn.safetensors` — основной текстовый энкодер (clip_name1), используется вместо штатного `gemma-3-12b-it-ablit-norms-biproj-fp8mixed.safetensors` из воркфлоу — тот падает с `AttributeError: 'Linear' object has no attribute 'weight'` при частичной загрузке на этом ComfyUI. Был на сервере до этой сессии, точный источник не сохранён.
- `ltx-2.3_text_projection_bf16.safetensors` (clip_name2) — источник не сохранён.
- `LTX23_audio_vae_bf16.safetensors` — источник не сохранён.
- `ltx-2.3-spatial-upscaler-x2-1.1.safetensors`, `taeltx2_3.safetensors`, `flownet.pkl` — см. `ltx_sulphur` выше, общие файлы.
- Параметризуется через env `LTX_EROS_CLIP_NAME1` в коде бота, если потребуется снова подменить энкодер.

### `mopmix` / `mopmix_duo` (workflow_mopmix.json, workflow_mopmix_duo.json)
- `mopMix-bigASP25_V10_High.safetensors`, `mopMix-bigASP25_V10_Low.safetensors` — чекпоинты (переименованы под имена, которые ждёт воркфлоу). Похожая модель найдена на CivitAI: **https://civitai.com/models/2128936** (версии "BigASP 2.5 High" id `2672542`, "BigASP 2.5 Low" id `2672758`, файлы `mopMix_bigasp25High.safetensors` / `...Low.safetensors`) — переименовать после скачивания. *Это переоткрытая, не оригинально сохранённая ссылка — сверить перед использованием.*
- `Eyeful_v2-Paired.pt` (детектор глаз для FaceDetailer) — Источник: **https://huggingface.co/Tenofas/ComfyUI** (папка `bbox/`).
- `face_yolov8m.pt` (детектор лица) — стандартная модель Impact-Pack, источник: **https://huggingface.co/Bingsu/adetailer** → `face_yolov8m.pt`, положить в `models/ultralytics/bbox/`.
- `sam_vit_b_01ec64.pth` (SAMLoader) — официальный чекпоинт Meta SAM: **https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth**, положить в `models/sams/`.
- Только `mopmix_duo`: ReActor —
  - `inswapper_128.onnx` (своп лица) — зеркалится в нескольких местах на HF/CivitAI из-за блокировки оригинала, см. инструкцию по установке в самом репо `https://github.com/Gourieff/ComfyUI-ReActor` (раздел Installation → models).
  - `retinaface_resnet50` (детекция лиц, facexlib) — скачивается автоматически при первом запуске ReActor.
  - `GFPGANv1.3.pth` (восстановление лица) — **https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth**
  - NSFW-фильтр ReActor (`scripts/reactor_sfw.py`) **пропатчен на сервере** (`nsfw_image()` всегда возвращает `False`) — без этого ReActor тихо выбрасывал кадры с высоким NSFW-score и отдавал почти чёрную картинку. При обновлении/переустановке ReActor патч слетит — нужно повторить.
- MopMix Duo также использует Florence-2 для описания внешности обоих людей (борется со
  смещением чекпоинта к "молодым спортивным" людям) — см. ниже, раздел Qwen/Florence.

### `image` (Edit Photo, Qwen-Image-Edit)
Собирается полностью в коде (`build_image_edit_workflow` в `telegram_comfyui_bot.py`), отдельного JSON-файла нет.
- `qwen_image_edit_2509_fp8_e4m3fn.safetensors` (UNET) — **https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI** → `split_files/diffusion_models/qwen_image_edit_2509_fp8_e4m3fn.safetensors`
- `qwen_2.5_vl_7b_fp8_scaled.safetensors` (CLIP, type=`qwen_image`) — **https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI** → `split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors`
- `qwen_image_vae.safetensors` (VAE) — тот же репозиторий → `split_files/vae/qwen_image_vae.safetensors`
- Опционально (сейчас не подключена в воркфлоу, но скачана на сервере про запас для ускорения) — `Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors`: **https://huggingface.co/lightx2v/Qwen-Image-Lightning** → `Qwen-Image-Edit-2509/Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors`

### Florence-2 (автоописание фото для MopMix Duo)
- `microsoft/Florence-2-large` — **https://huggingface.co/microsoft/Florence-2-large** (скачивается нодой `DownloadAndLoadFlorence2Model` автоматически при первом запуске, кладёт в `models/LLM/Florence-2-large`).

## Прочее окружение

- `deep-translator` (pip) — перевод промта RU→EN для MopMix/MopMix Duo (их SDXL CLIP-энкодер не понимает русский; LTX Sulphur/Eros используют мультиязычные LLM-энкодеры и в переводе не нуждаются).
- `word2number` (pip) — зависимость `ComfyUI-PromptRelay`.
- Переменные окружения бота — см. `os.getenv(...)` по всему `telegram_comfyui_bot.py` (пути к воркфлоу, ReActor-настройки, denoise для MopMix/Duo, качество LTX и т.д.) — у каждой есть разумный дефолт в коде.

## Известные особенности / не баги

- `LTX2.3_reasoning_I2V_V3.safetensors`, `LTX2.3_Multi_step_video_reasoning_V0.1.safetensors`,
  `DaSiWa_LTX23_NSFW_Bodyphysics_Fluid_Motion_Enhancer_v01.safetensors`, `OmniNFT_converted_lora.safetensors`
  (переименован из `LTX-2.3-OmniNFT-RL-Lora_bf16.safetensors` авторства kijai) — скачаны по прямым ссылкам,
  присланным пользователем в чате; сами ссылки не сохранены в этой сессии — при потере файла придётся искать заново.
- `gonzaLomo_DMD_v30.json` и чекпоинт `gonzalomoXLFluxPony_v50FluXLDMD.safetensors`
  (**https://civitai.com/models/1513492**, версия `v5.0 FluXL DMD` id `2052057`) — режим GonzaLomo
  был добавлен и затем убран из бота по решению пользователя; файлы оставлены на диске, но
  бот их больше не использует.
