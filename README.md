# AutoFisher

Windows WASAPI fishing splash detector.

The splash templates are stored in [sounds](./sounds).

## Setup

Requires Python 3.14 and [uv](https://docs.astral.sh/uv/).

```powershell
uv sync
uv run autofisher
```

## Build a Windows executable

```powershell
uv sync
make build
```

The standalone executable is generated in `dist/AutoFisher.exe`.

## Development

```powershell
make sync
make test
```

Install and run the pre-commit hooks:

```powershell
uv run pre-commit install
uv run pre-commit run --all-files
```
