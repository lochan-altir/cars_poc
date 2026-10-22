# Project explanation

`cars_poc` is a Python proof of concept that turns vehicle images into structured
metadata using vision-capable language models. It is intended for car listings
and automotive advertising: besides identifying a vehicle, it describes the
setting, photography, creative direction, and image format.

## Two ways to use the project

- **Streamlit UI (`app.py`):** upload an image, select a provider and model, and
  view the resulting JSON. The UI calls providers directly and does not require
  the FastAPI service. It accepts PNG, JPG/JPEG, WebP, and GIF uploads, then uses
  Pillow to encode the opened image as PNG or JPEG for inference.
- **FastAPI service (`car_classifier/`):** exposes multipart image-upload endpoints
  for integrations. Each provider route parses and normalizes the model output,
  then validates it with the shared Pydantic response schema.

Both paths support Groq, Ollama, OpenAI, and Anthropic. Hosted providers require
their corresponding API credentials; Ollama requires a reachable server with a
compatible vision model installed. Model choices in the source are configured
options, not a guarantee of current provider availability.

## Classification workflow

1. The user uploads an image through the UI or a REST endpoint.
2. The application sends the image and a classification prompt to the selected
   provider. Hosted requests carry a base64-encoded image; the API's Ollama route
   passes image bytes through the Ollama client.
3. The model is asked to return JSON describing the vehicle and its surroundings.
4. Parsing helpers attempt to recover object data from responses containing code
   fences, extra text, malformed escapes, or Python-style dictionary literals.
5. Normalization fills missing metadata such as the filename and timestamp, and
   coerces inconsistent strings, booleans, and lists.
6. The UI displays the normalized dictionary. The API additionally validates it
   as `CarClassificationResponse` before returning it.

## What the output contains

The response includes `image_id`, `source_file`, `classified_at`, a model-provided
`confidence_score`, and a flat `raw_labels` list. Six nested sections organize the
classification:

| Section | Examples of information |
| --- | --- |
| `vehicle` | Brand, model, category, trim, color, view angle, vehicle count |
| `location` | Environment, terrain, road type, inferred region, background |
| `scene` | Time of day, season, weather, sky prominence |
| `visual` | Palette, dominant hues, saturation, lighting, shot type, motion |
| `creative` | Lifestyle segment, emotional tone, campaign theme, people, activity |
| `format` | Ad format, aspect ratio, orientation, copy and visible branding |

The authoritative API field definitions are in
[`car_classifier/models/schemas.py`](car_classifier/models/schemas.py). Many nested
fields are optional strings; the prompt's suggested categories are not all
enforced as enums by the schema.

## Code organization

| File or directory | Responsibility |
| --- | --- |
| `app.py` | Standalone UI, provider calls, prompt, parsing, and normalization |
| `car_classifier/main.py` | FastAPI app, CORS configuration, routers, health endpoint |
| `car_classifier/config.py` | API configuration and provider/model settings |
| `car_classifier/routers/` | Provider-specific request and response handling |
| `car_classifier/core/prompt.py` | API classification prompt |
| `car_classifier/core/json_utils.py` | Lenient JSON parsing for hosted API routes |
| `car_classifier/core/image_utils.py` | Uploaded-file base64 conversion |
| `car_classifier/models/schemas.py` | Pydantic request/response models and model choices |
| `requirements.txt` | Python dependencies |
| `Makefile` | Environment installation and UI/API launch commands |
| `Dockerfile` | Python 3.11 image running the FastAPI service on port 8000 |

The Streamlit app duplicates prompt and processing logic from the API package.
API normalization is currently implemented in `routers/ollama.py` and reused by
the other provider routers. Changes to classification behavior may therefore
need corresponding updates in both execution paths.

## Running locally

From the repository root, create the virtual environment and install dependencies:

```bash
make install
```

Configure the selected provider locally as described in [README.md](README.md).
The Streamlit app loads a `.env` file from the project root, or its parent if no
root file exists. Keep credentials out of version control.

Start the browser UI:

```bash
make run-app
```

Alternatively, start the REST service:

```bash
make run-backend
```

The API's interactive documentation is at `http://127.0.0.1:8000/docs`.

| Method | Route | Input or purpose |
| --- | --- | --- |
| GET | `/health` | Returns `{"status": "ok"}`; does not check provider connectivity |
| POST | `/classify/groq` | Multipart `image`; model chosen by configuration |
| POST | `/classify/ollama` | Multipart `image`; configured model with installed-model fallback |
| POST | `/classify/openai` | Multipart `image` and required `model` form field |
| POST | `/classify/anthropic` | Multipart `image` and required `model` form field |

Use `/docs` to see the allowed model values. The Docker build and run commands
are documented in the README; the container runs the API only.

## Current scope and limitations

This is an inference application, with no implemented database, training pipeline,
or classification history. Uploaded images are sent to the selected provider;
Ollama uses the configured server, which may be local or remote.

Descriptions such as trim, region, and campaign intent are model inferences and
may be inaccurate. The confidence value comes from the model and is not an
independently calibrated accuracy measure. Normalization improves structural
consistency but does not verify whether the descriptions are correct.

The prompt requests an error object for non-vehicle images, but neither path
explicitly handles that object before normalization. A clean rejection of such
images is therefore not guaranteed. The UI also skips the API's Pydantic response
validation, so the two entry points can produce different results for malformed
model output.

The API currently allows all CORS origins and defines no authentication layer.
Provider failures, unavailable models, and invalid model responses can prevent
classification. The repository does not currently include an automated test suite.
