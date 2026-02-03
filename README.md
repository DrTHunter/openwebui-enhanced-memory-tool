# OpenWebUI Enhanced Memory Tool

Enhanced memory tool for **Open WebUI** that adds **categories + tags**, **batch operations**, **smart duplicate detection**, and **content-based search** for stored memories. Built to improve long-term recall quality while keeping clear boundaries around what should and should not be saved.

## File

- `openwebui_memory_tool_enhanced.py`

## Key Features

- **Categorized memories** with optional **tags**
- **Duplicate detection** using text similarity (configurable threshold)
- **Batch operations** (add/update/delete) for better performance
- **Search** memories by content (and optionally category/tags)
- Stores metadata (created timestamp, version) when categories are enabled

## Safety Rules (Recommended)

### SHOULD store
- Stable personal info (name, job, long-term preferences)
- Ongoing projects, goals, recurring issues
- Strong likes/dislikes, constraints, boundaries

### SHOULD NOT store
- Passwords, secrets, logins, tokens
- Highly sensitive identifiers (SSNs, full address, etc.)

## Installation (Open WebUI)

1. Copy `openwebui_memory_tool_enhanced.py` into your **Open WebUI workspace Tools directory** (the “Tools” section location in your Open WebUI workspace).
2. Reload/restart Open WebUI so it detects the new tool.
3. Enable the tool in the Open WebUI Tools UI (if your setup requires manual enabling).

## Configuration (Valves)

Inside the tool, settings live under `Tools.Valves`:

- `USE_MEMORY` (default `True`) — enable/disable memory usage
- `DEBUG` (default `True`) — verbose debug behavior
- `SIMILARITY_THRESHOLD` (default `0.85`) — sensitivity for duplicate detection (0–1)
- `ENABLE_CATEGORIES` (default `True`) — store structured JSON (category/tags/metadata)
- `DEFAULT_CATEGORY` (default `"general"`) — category used when none is provided

## Stored Memory Format

When categorization is enabled, memories are stored as JSON:

```json
{
  "text": "User prefers dark mode.",
  "category": "preferences",
  "tags": ["ui", "accessibility"],
  "created": "2026-02-03T12:34:56.000000",
  "version": "2.0"
}
```

Older/plaintext entries are treated as **legacy** format automatically.

## Tool Methods (What it Provides)

### Recall memories
Retrieve all memories, or filter by category/tags.

- `recall_memories(user, category=None, tags=None)`

### Add memory (with duplicate detection)
Add one or many entries, skipping near-duplicates automatically.

- `add_memory(input_text, user, category=None, tags=None)`

### Search memories
Search across text/category/tags with relevance scoring.

- `search_memories(query, user, category=None, search_fields=None)`

### Batch operations
Run multiple operations in one call:

Supported types:
- `add`
- `update`
- `delete`

Example operations payload:

```json
[
  {"type": "add", "data": {"text": "Working on Orion Forge", "category": "projects", "tags": ["orion", "forge"]}},
  {"type": "update", "data": {"index": 2, "text": "Updated memory text"}},
  {"type": "delete", "data": {"index": 3}}
]
```

Method:
- `batch_operations(operations, user)`

### Update memory
Updates memory by index while attempting to preserve metadata.

- `update_memory(updates, user)`

Where `updates` looks like:

```python
[{"index": 1, "content": "Replacement text"}]
```

### Delete memory
Deletes memories by index:

- `delete_memory(indices, user)`

## Notes / Known Limitations

- Duplicate detection is **string similarity**, not embedding-based semantic similarity.
- Batch operations call individual operations internally; it reduces overhead but still depends on backend memory performance.
- Tag filtering in recall is currently “match any tag” logic.

## License

MIT
