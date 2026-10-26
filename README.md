# Team Churn API

Team Churn API is a prediction product for telecom customer churn. It exposes a
validated HTTP interface for submitting customer usage and account details and
receiving a churn classification with its estimated probability.

The service runs a trained scikit-learn pipeline packaged as
`churn_pipeline.joblib`. Its API translates snake_case request fields into the
exact feature names and order expected by the model, and is available on port
`8080` by default.

## Repository Structure

```text
.
├── api/
│   └── main.py
├── ml_platform/
│   └── training/
│       └── churn/
│           ├── train.py
│           └── models/
│               ├── churn_pipeline.joblib
│               └── feature_names.json
├── production_app/
│   └── tests/
│       └── test_api.py
├── Dockerfile
├── entrypoint.sh
├── pyproject.toml
└── uv.lock
```

## Getting Started and Local Setup

The project requires Python 3.12 or newer and uses `uv` to manage dependencies.
Install `uv` by following the [official installation guide](https://docs.astral.sh/uv/getting-started/installation/).
Runtime dependencies include FastAPI, Uvicorn, Pydantic, pandas, scikit-learn,
and joblib. The development dependency group includes Ruff, pytest, and HTTPX.
Exact resolved versions are recorded in `uv.lock`.

From the repository root, synchronize the locked dependencies:

```bash
uv sync --locked
```

Start the API locally:

```bash
uv run uvicorn api.main:app --host 0.0.0.0 --port 8080
```

The service is then available at `http://localhost:8080`. The model artifact and
`feature_names.json` must be present at their repository paths before the
application starts.

## API Endpoints

### `GET /health`

Returns the service health status and identifying information.

Example response:

```json
{
  "status": "ok",
  "service": "telecom-churn-api",
  "version": "v2"
}
```

### `POST /predictions`

Accepts a JSON object containing all 13 model inputs. Fields are validated by
Pydantic before inference. Numeric fields must be non-negative except
`age_group`, `tariff_plan`, and `status`, which must be at least `1`.
`complains` must be either `0` or `1`.

Example request:

```json
{
  "call_failure": 8,
  "complains": 0,
  "subscription_length": 24,
  "charge_amount": 15,
  "seconds_of_use": 1200,
  "frequency_of_use": 30,
  "frequency_of_sms": 10,
  "distinct_called_numbers": 12,
  "age_group": 3,
  "tariff_plan": 1,
  "status": 1,
  "age": 35,
  "customer_value": 120.5
}
```

Each request field maps to the corresponding model feature name, preserving the
model's exact spelling and feature order:

| Request field | Model feature |
| --- | --- |
| `call_failure` | `Call  Failure` |
| `complains` | `Complains` |
| `subscription_length` | `Subscription  Length` |
| `charge_amount` | `Charge  Amount` |
| `seconds_of_use` | `Seconds of Use` |
| `frequency_of_use` | `Frequency of use` |
| `frequency_of_sms` | `Frequency of SMS` |
| `distinct_called_numbers` | `Distinct Called Numbers` |
| `age_group` | `Age Group` |
| `tariff_plan` | `Tariff Plan` |
| `status` | `Status` |
| `age` | `Age` |
| `customer_value` | `Customer Value` |

The mapping is defined in `api/main.py`, and the ordered feature names are
stored in `ml_platform/training/churn/models/feature_names.json`.

Example response:

```json
{
  "prediction": 0,
  "churn_probability": 0.1234,
  "model_version": "week1-v1"
}
```

The `prediction` value is the model's predicted class, and
`churn_probability` is the predicted probability for class `1`, rounded to
four decimal places. Invalid requests return a validation error.

Interactive API documentation is available at
[`http://localhost:8080/docs`](http://localhost:8080/docs).

## Docker

Pre-built Docker images are available on GitHub Container Registry (GHCR).

**Published images:**

- [`v1`](https://github.com/ReginaldErzoah/team-churn-api/pkgs/container/team-churn-api/versions) — `ghcr.io/reginalderzoah/team-churn-api:v1`
- [`v2`](https://github.com/ReginaldErzoah/team-churn-api/pkgs/container/team-churn-api/versions) — `ghcr.io/reginalderzoah/team-churn-api:v2`

### Run a published image

Pull the latest published version (`v2`):

```bash
docker pull ghcr.io/reginalderzoah/team-churn-api:v2
```

Run the container:

```bash
docker run --rm -p 8080:8080 ghcr.io/reginalderzoah/team-churn-api:v2
```

The API will be available at:

- API: http://localhost:8080
- Interactive API documentation: http://localhost:8080/docs
- Health check: http://localhost:8080/health

To use version `v1`, replace `v2` with `v1` in the commands above.

### Build the image locally

Alternatively, build the image from the repository root:

```bash
docker build -t team-churn:v1 .
```

Run the locally built image:

```bash
docker run --rm -p 8080:8080 team-churn:v1
```

The container entrypoint starts Uvicorn on port `8080` by default and honors the `PORT` environment variable if a different listening port is needed.
