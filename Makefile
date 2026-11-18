.PHONY: sync test build clean run

UV := uv
PYTHON := $(UV) run python
RUFF := $(UV) run ruff
PYREFLY := $(UV) run pyrefly
PYTEST := $(UV) run pytest
PRE_COMMIT := $(UV) run pre-commit
PYINSTALLER := $(UV) run pyinstaller

ifeq ($(OS),Windows_NT)
DATA_SEPARATOR := ;
CLEAN_COMMAND := if exist build rmdir /s /q build & if exist dist rmdir /s /q dist
else
DATA_SEPARATOR := :
CLEAN_COMMAND := rm -rf build dist
endif

sync:
	$(UV) sync --python 3.14

test:
	$(RUFF) format --check .
	$(RUFF) check .
	$(PYREFLY) check
	$(PRE_COMMIT) run --all-files
	$(PYTEST) --cov-fail-under=0

build:
	$(PYINSTALLER) --clean --noconfirm --onefile --name AutoFisher --add-data "sounds$(DATA_SEPARATOR)sounds" --collect-all pyaudiowpatch --collect-all pyautogui build_exe.py

run:
	$(UV) run autofisher

clean:
	$(CLEAN_COMMAND)
