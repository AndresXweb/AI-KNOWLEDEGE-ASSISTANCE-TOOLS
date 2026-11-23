
import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError


IGNORED_DIRS = {
    ".git", ".venv", "venv", "env", "node_modules",
    "__pycache__", ".idea", ".vscode", "dist", "build",
    ".next", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "data",
}

SAFE_ENV_FILES = {".env.example", ".env.sample", ".env.template"}

SENSITIVE_SUFFIXES = {
    ".pem", ".key", ".pfx", ".p12",
    ".db", ".sqlite", ".sqlite3",
}

MAX_FILE_BYTES = 1_000_000
MAX_LIST_ENTRIES = 200
MAX_READ_LINES = 300
MAX_READ_CHARS = 20_000
MAX_SEARCH_RESULTS = 40
MAX_SEARCH_FILES = 2000


class ToolError(Exception):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


class ListFilesArgs(BaseModel):
    directory: str = Field(
        default=".",
        description="Carpeta relativa a la raíz del repositorio",
    )
    max_depth: int = Field(default=3, ge=1, le=5)


class ReadFileArgs(BaseModel):
    path: str = Field(
        min_length=1,
        description="Ruta relativa del archivo",
    )
    start_line: int = Field(default=1, ge=1)
    end_line: int | None = Field(default=None, ge=1)


class SearchTextArgs(BaseModel):
    query: str = Field(min_length=2, max_length=100)


def _is_ignored_dir(name: str) -> bool:
    return name in IGNORED_DIRS or name.endswith(".egg-info")


def _is_sensitive_file(name: str) -> bool:
    lower = name.lower()

    if lower.startswith(".env"):
        return lower not in SAFE_ENV_FILES

    if lower.startswith(("id_rsa", "id_ed25519")):
        return True

    return Path(lower).suffix in SENSITIVE_SUFFIXES


def _looks_binary(path: Path) -> bool:
    with path.open("rb") as handle:
        return b"\0" in handle.read(2048)


class RepoToolset:
    """Herramientas de SOLO LECTURA sobre un repositorio local.

    Nunca salen de `root`, ignoran carpetas pesadas o privadas
    (.git, .venv, node_modules...) y bloquean archivos
    sensibles (.env, llaves, bases de datos).
    """

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()

        self._tools = {
            "list_files": (
                ListFilesArgs,
                self._list_files,
                "Lista archivos y carpetas del repositorio del "
                "proyecto. Úsala primero para conocer la "
                "estructura.",
            ),
            "read_file": (
                ReadFileArgs,
                self._read_file,
                "Lee un archivo de texto del repositorio "
                "(máximo 300 líneas por llamada; usa "
                "start_line y end_line para leer más).",
            ),
            "search_text": (
                SearchTextArgs,
                self._search_text,
                "Busca un texto (sin distinguir mayúsculas) en "
                "todos los archivos del repositorio y devuelve "
                "archivo, línea y contenido.",
            ),
        }

    # ---------- interfaz para el agente ----------

    @property
    def names(self) -> set[str]:
        return set(self._tools)

    def definitions(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": model.model_json_schema(),
                },
            }
            for name, (model, _fn, description)
            in self._tools.items()
        ]

    def execute(
        self, name: str, raw_arguments: str
    ) -> dict[str, Any]:
        entry = self._tools.get(name)

        if entry is None:
            return {"ok": False, "error": "unknown_tool"}

        model, function, _description = entry

        try:
            args = model.model_validate(json.loads(raw_arguments))
        except (json.JSONDecodeError, ValidationError, ValueError):
            return {"ok": False, "error": "invalid_arguments"}

        try:
            return {"ok": True, "data": function(args)}
        except ToolError as exc:
            return {"ok": False, "error": exc.code}

    # ---------- seguridad de rutas ----------

    def _resolve(self, relative: str) -> Path:
        path = (self.root / relative).resolve()

        if not path.is_relative_to(self.root):
            raise ToolError("path_not_allowed")

        parts = path.relative_to(self.root).parts

        if any(_is_ignored_dir(p) for p in parts[:-1]):
            raise ToolError("path_not_allowed")

        if parts:
            last = parts[-1]

            if path.is_dir():
                if _is_ignored_dir(last):
                    raise ToolError("path_not_allowed")
            elif _is_sensitive_file(last) or _is_ignored_dir(last):
                raise ToolError("path_not_allowed")

        return path

    def _rel(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()

    def _walk_files(self, base: Path):
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = sorted(
                d for d in dirnames
                if not _is_ignored_dir(d)
                and not os.path.islink(os.path.join(dirpath, d))
            )

            for filename in sorted(filenames):
                full = Path(dirpath) / filename

                if (
                    _is_sensitive_file(filename)
                    or full.is_symlink()
                ):
                    continue

                yield full

    # ---------- herramientas ----------

    def _list_files(self, args: ListFilesArgs) -> dict:
        base = self._resolve(args.directory)

        if not base.is_dir():
            raise ToolError("not_found")

        entries: list[str] = []

        for dirpath, dirnames, filenames in os.walk(base):
            depth = len(
                Path(dirpath).relative_to(base).parts
            ) + 1

            dirnames[:] = sorted(
                d for d in dirnames
                if not _is_ignored_dir(d)
                and not os.path.islink(os.path.join(dirpath, d))
            )

            if depth > args.max_depth:
                dirnames[:] = []
                continue

            for d in dirnames:
                entries.append(
                    self._rel(Path(dirpath) / d) + "/"
                )

            for f in sorted(filenames):
                if not _is_sensitive_file(f):
                    entries.append(
                        self._rel(Path(dirpath) / f)
                    )

        entries.sort()
        truncated = len(entries) > MAX_LIST_ENTRIES

        return {
            "directory": self._rel(base) or ".",
            "entries": entries[:MAX_LIST_ENTRIES],
            "truncated": truncated,
        }

    def _read_file(self, args: ReadFileArgs) -> dict:
        path = self._resolve(args.path)

        if not path.is_file():
            raise ToolError("not_found")

        if path.stat().st_size > MAX_FILE_BYTES:
            raise ToolError("file_too_large")

        if _looks_binary(path):
            raise ToolError("binary_file")

        lines = path.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()

        total = len(lines)
        start = args.start_line
        end = min(
            args.end_line or total,
            total,
            start + MAX_READ_LINES - 1,
        )

        selected = lines[start - 1:end]
        content = "\n".join(
            f"{start + i}: {line}"
            for i, line in enumerate(selected)
        )

        truncated = end < (args.end_line or total)

        if len(content) > MAX_READ_CHARS:
            content = content[:MAX_READ_CHARS]
            truncated = True

        return {
            "path": self._rel(path),
            "start_line": start,
            "end_line": end,
            "total_lines": total,
            "content": content,
            "truncated": truncated,
        }

    def _search_text(self, args: SearchTextArgs) -> dict:
        needle = args.query.lower()
        results: list[dict[str, Any]] = []
        scanned = 0

        for path in self._walk_files(self.root):
            scanned += 1

            if scanned > MAX_SEARCH_FILES:
                break

            try:
                if (
                    path.stat().st_size > MAX_FILE_BYTES
                    or _looks_binary(path)
                ):
                    continue

                text = path.read_text(
                    encoding="utf-8", errors="replace"
                )
            except OSError:
                continue

            for number, line in enumerate(
                text.splitlines(), start=1
            ):
                if needle in line.lower():
                    results.append({
                        "path": self._rel(path),
                        "line": number,
                        "text": line.strip()[:200],
                    })

                    if len(results) >= MAX_SEARCH_RESULTS:
                        return {
                            "results": results,
                            "truncated": True,
                        }

        return {"results": results, "truncated": False}
