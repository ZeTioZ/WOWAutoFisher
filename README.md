# AutoFisher

Windows WASAPI fishing splash detector.

The splash templates are stored in [sounds](./sounds).

## Setup

Requires Python 3.14 and [uv](https://docs.astral.sh/uv/).

Before running AutoFisher, install
[Better Fishing Forever](https://www.curseforge.com/wow/addons/better-fishing-forever)
in World of Warcraft and bind its **Full Cast** action to `R`. AutoFisher presses `R`
to retrieve and recast, so it will not work correctly with a different key binding.

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
