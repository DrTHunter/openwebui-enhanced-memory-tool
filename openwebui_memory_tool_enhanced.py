"""
title: Enhanced Memory Enhancement Tool for LLM Web UI
author: Trent Hunter
version: 1.0
license: MIT

Enhanced with:
- Memory categories and tagging
- Batch operations for better performance
- Smart duplicate detection
- Content-based search capabilities

The assistant SHOULD call this tool when the user shares:
- stable personal info (name, age, job, long-term preferences)
- ongoing projects, goals, or recurring issues
- strong likes/dislikes, constraints, or boundaries

The assistant SHOULD NOT store:
- passwords, secrets, logins, tokens
- highly sensitive identifiers (SSNs, full address, etc.)
"""

import json
import re
from typing import Callable, Any, List, Dict, Union, Optional
from datetime import datetime
from difflib import SequenceMatcher
import numpy as np
from collections import defaultdict

from open_webui.models.memories import Memories
from pydantic import BaseModel, Field


class EventEmitter:
    def __init__(self, event_emitter: Callable[[dict], Any] = None):
        self.event_emitter = event_emitter

    async def emit(self, description="Unknown state", status="in_progress", done=False):
        """
        Send a status event to the event emitter.

        :param description: Event description
        :param status: Event status
        :param done: Whether the event is complete
        """
        if self.event_emitter:
            await self.event_emitter(
                {
                    "type": "status",
                    "data": {
                        "status": status,
                        "description": description,
                        "done": done,
                    },
                }
            )


class TextSimilarity:
    """Text similarity detection using multiple methods"""

    @staticmethod
    def similarity(a: str, b: str) -> float:
        """Calculate similarity between two strings (0-1)"""
        return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()

    @staticmethod
    def find_similar_memories(
        new_text: str, existing_memories: List, threshold: float = 0.8
    ) -> List:
        """Find memories similar to the new text"""
        similar = []
        for memory in existing_memories:
            # Extract content from memory object
            content = memory.content
            # If content is JSON, parse it to get the actual text
            if content.strip().startswith("{"):
                try:
                    data = json.loads(content)
                    content = data.get("text", content)
                except:
                    pass

            similarity_score = TextSimilarity.similarity(new_text, content)
            if similarity_score >= threshold:
                similar.append((memory, similarity_score))

        return sorted(similar, key=lambda x: x[1], reverse=True)


class MemorySearch:
    """Enhanced memory search functionality"""

    @staticmethod
    def search_memories(
        memories: List, query: str, search_fields: List[str] = None
    ) -> List:
        """Search memories by content with flexible matching"""
        if not search_fields:
            search_fields = ["text", "category", "tags"]

        results = []
        query_terms = query.lower().split()

        for memory in memories:
            content = memory.content
            memory_data = {}

            # Parse memory content (could be plain text or JSON)
            if content.strip().startswith("{"):
                try:
                    memory_data = json.loads(content)
                except:
                    memory_data = {"text": content}
            else:
                memory_data = {"text": content}

            # Calculate match score
            score = 0
            for field in search_fields:
                if field in memory_data and memory_data[field]:
                    field_content = memory_data[field]
                    if isinstance(field_content, list):
                        field_content = " ".join(field_content)
                    field_content = field_content.lower()

                    # Exact phrase match
                    if query.lower() in field_content:
                        score += 3

                    # Individual term matches
                    for term in query_terms:
                        if term in field_content:
                            score += 1

            if score > 0:
                results.append((memory, score))

        return sorted(results, key=lambda x: x[1], reverse=True)


class Tools:
    class Valves(BaseModel):
        USE_MEMORY: bool = Field(
            default=True, description="Enable or disable memory usage."
        )
        DEBUG: bool = Field(default=True, description="Enable or disable debug mode.")
        SIMILARITY_THRESHOLD: float = Field(
            default=0.85, description="Threshold for duplicate detection (0-1)"
        )
        ENABLE_CATEGORIES: bool = Field(
            default=True, description="Enable memory categorization"
        )
        DEFAULT_CATEGORY: str = Field(
            default="general", description="Default category for memories"
        )

    def __init__(self):
        self.valves = self.Valves()

    def _validate_user(self, __user__: dict) -> tuple[bool, str]:
        """Validate user data and return (is_valid, user_id)"""
        if not __user__:
            return False, "User dictionary not provided"

        user_id = __user__.get("id")
        if not user_id:
            return False, "User ID not found in user dictionary"

        return True, user_id

    def _parse_memory_content(self, memory) -> Dict:
        """Parse memory content whether it's JSON or plain text"""
        content = memory.content
        if content.strip().startswith("{"):
            try:
                return json.loads(content)
            except:
                return {"text": content, "category": "legacy"}
        else:
            return {"text": content, "category": "legacy"}

    def _format_memory_content(
        self, text: str, category: str = None, tags: List[str] = None
    ) -> str:
        """Format memory content with metadata"""
        if self.valves.ENABLE_CATEGORIES:
            memory_data = {
                "text": text.strip(),
                "category": category or self.valves.DEFAULT_CATEGORY,
                "tags": tags or [],
                "created": datetime.now().isoformat(),
                "version": "2.0",
            }
            return json.dumps(memory_data, ensure_ascii=False)
        else:
            return text.strip()

    async def recall_memories(
        self,
        __user__: dict = None,
        __event_emitter__: Callable[[dict], Any] = None,
        category: str = None,
        tags: List[str] = None,
    ) -> str:
        """
        Retrieves stored memories with optional filtering by category and tags.

        :param category: Filter by category (optional)
        :param tags: Filter by tags (optional)
        :return: Formatted list of memories
        """
        emitter = EventEmitter(__event_emitter__)

        is_valid, user_id = self._validate_user(__user__)
        if not is_valid:
            await emitter.emit(description=user_id, status="missing_user_id", done=True)
            return json.dumps({"message": user_id}, ensure_ascii=False)

        await emitter.emit(
            description="Retrieving stored memories.",
            status="recall_in_progress",
            done=False,
        )

        user_memories = Memories.get_memories_by_user_id(user_id)
        if not user_memories:
            message = "No memory stored."
            await emitter.emit(description=message, status="recall_complete", done=True)
            return json.dumps({"message": message}, ensure_ascii=False)

        # Filter memories if category or tags specified
        filtered_memories = []
        for memory in user_memories:
            memory_data = self._parse_memory_content(memory)

            include = True
            if category and memory_data.get("category") != category:
                include = False
            if tags and not any(tag in memory_data.get("tags", []) for tag in tags):
                include = False

            if include:
                filtered_memories.append(memory)

        if not filtered_memories:
            filter_msg = ""
            if category:
                filter_msg += f" in category '{category}'"
            if tags:
                filter_msg += f" with tags {tags}"
            message = f"No memories found{filter_msg}."
            await emitter.emit(description=message, status="recall_complete", done=True)
            return json.dumps({"message": message}, ensure_ascii=False)

        # Format output
        content_list = []
        for index, memory in enumerate(
            sorted(filtered_memories, key=lambda m: m.created_at), start=1
        ):
            memory_data = self._parse_memory_content(memory)
            base_text = f"{index}. {memory_data['text']}"

            # Add metadata if available
            meta_parts = []
            if memory_data.get("category") and memory_data["category"] != "legacy":
                meta_parts.append(f"Category: {memory_data['category']}")
            if memory_data.get("tags"):
                meta_parts.append(f"Tags: {', '.join(memory_data['tags'])}")

            if meta_parts:
                base_text += f" [{'; '.join(meta_parts)}]"

            content_list.append(base_text)

        await emitter.emit(
            description=f"{len(filtered_memories)} memories loaded",
            status="recall_complete",
            done=True,
        )

        return f"Memories from the user's memory vault: {content_list}"

    async def add_memory(
        self,
        input_text: List[str],
        __user__: dict = None,
        __event_emitter__: Callable[[dict], Any] = None,
        category: str = None,
        tags: List[str] = None,
    ) -> str:
        """
        Add new memories with duplicate detection and categorization.

        :param input_text: Text to add to memory
        :param category: Optional category for the memory
        :param tags: Optional tags for the memory
        :return: Result message
        """
        emitter = EventEmitter(__event_emitter__)
        is_valid, user_id = self._validate_user(__user__)
        if not is_valid:
            await emitter.emit(description=user_id, status="missing_user_id", done=True)
            return json.dumps({"message": user_id}, ensure_ascii=False)

        if isinstance(input_text, str):
            input_text = [input_text]

        await emitter.emit(
            description="Adding entries to memory vault with duplicate check.",
            status="add_in_progress",
            done=False,
        )

        # Get existing memories for duplicate check
        existing_memories = Memories.get_memories_by_user_id(user_id)

        added_items = []
        duplicate_items = []
        failed_items = []

        for item in input_text:
            if not item or not item.strip():
                continue

            # Check for duplicates
            similar_memories = TextSimilarity.find_similar_memories(
                item.strip(), existing_memories, self.valves.SIMILARITY_THRESHOLD
            )

            if similar_memories:
                duplicate_items.append(item.strip())
                continue

            # Format and store memory
            formatted_content = self._format_memory_content(item, category, tags)
            new_memory = Memories.insert_new_memory(user_id, formatted_content)

            if new_memory:
                added_items.append(item.strip())
                # Add to existing memories for subsequent duplicate checks
                existing_memories.append(new_memory)
            else:
                failed_items.append(item.strip())

        # Build result message
        messages = []
        if added_items:
            messages.append(f"Added {len(added_items)} memories")
        if duplicate_items:
            messages.append(f"Skipped {len(duplicate_items)} duplicates")
        if failed_items:
            messages.append(f"Failed to add {len(failed_items)} memories")

        result_message = ". ".join(messages) if messages else "No memories processed"

        await emitter.emit(
            description=result_message,
            status="add_complete",
            done=True,
        )
        return json.dumps({"message": result_message}, ensure_ascii=False)

    async def batch_operations(
        self,
        operations: List[Dict],
        __user__: dict = None,
        __event_emitter__: Callable[[dict], Any] = None,
    ) -> str:
        """
        Perform multiple memory operations in a single batch.

        :param operations: List of operation dictionaries
        Each operation should have:
          - 'type': 'add', 'delete', or 'update'
          - 'data': Operation-specific data
        :return: Batch operation results
        """
        emitter = EventEmitter(__event_emitter__)
        is_valid, user_id = self._validate_user(__user__)
        if not is_valid:
            return json.dumps({"message": user_id}, ensure_ascii=False)

        await emitter.emit(
            description=f"Processing {len(operations)} batch operations",
            status="batch_in_progress",
            done=False,
        )

        results = []
        for i, op in enumerate(operations):
            op_type = op.get("type")
            op_data = op.get("data", {})

            try:
                if op_type == "add":
                    result = await self.add_memory(
                        [op_data.get("text", "")],
                        __user__,
                        None,  # Don't emit for individual ops
                        op_data.get("category"),
                        op_data.get("tags", []),
                    )
                elif op_type == "delete":
                    result = await self.delete_memory(
                        [op_data.get("index")], __user__, None
                    )
                elif op_type == "update":
                    result = await self.update_memory(
                        [
                            {
                                "index": op_data.get("index"),
                                "content": op_data.get("text", ""),
                            }
                        ],
                        __user__,
                        None,
                    )
                else:
                    result = json.dumps(
                        {"message": f"Unknown operation type: {op_type}"}
                    )

                results.append(
                    {
                        "operation": i + 1,
                        "type": op_type,
                        "success": "error" not in json.loads(result).get("message", ""),
                        "result": json.loads(result),
                    }
                )

            except Exception as e:
                results.append(
                    {
                        "operation": i + 1,
                        "type": op_type,
                        "success": False,
                        "error": str(e),
                    }
                )

        await emitter.emit(
            description=f"Completed {len(operations)} batch operations",
            status="batch_complete",
            done=True,
        )

        return json.dumps({"batch_results": results}, ensure_ascii=False)

    async def search_memories(
        self,
        query: str,
        __user__: dict = None,
        __event_emitter__: Callable[[dict], Any] = None,
        category: str = None,
        search_fields: List[str] = None,
    ) -> str:
        """
        Search memories by content with flexible matching.

        :param query: Search query string
        :param category: Optional category filter
        :param search_fields: Fields to search in ['text', 'category', 'tags']
        :return: Search results
        """
        emitter = EventEmitter(__event_emitter__)
        is_valid, user_id = self._validate_user(__user__)
        if not is_valid:
            await emitter.emit(description=user_id, status="missing_user_id", done=True)
            return json.dumps({"message": user_id}, ensure_ascii=False)

        await emitter.emit(
            description=f"Searching memories for: {query}",
            status="search_in_progress",
            done=False,
        )

        user_memories = Memories.get_memories_by_user_id(user_id)
        if not user_memories:
            message = "No memories available to search."
            await emitter.emit(description=message, status="search_complete", done=True)
            return json.dumps({"message": message}, ensure_ascii=False)

        # Filter by category first if specified
        if category:
            filtered_memories = []
            for memory in user_memories:
                memory_data = self._parse_memory_content(memory)
                if memory_data.get("category") == category:
                    filtered_memories.append(memory)
            user_memories = filtered_memories

        # Perform search
        search_results = MemorySearch.search_memories(
            user_memories, query, search_fields
        )

        if not search_results:
            message = f"No memories found matching '{query}'"
            if category:
                message += f" in category '{category}'"
            await emitter.emit(description=message, status="search_complete", done=True)
            return json.dumps({"message": message}, ensure_ascii=False)

        # Format results
        content_list = []
        for index, (memory, score) in enumerate(search_results, start=1):
            memory_data = self._parse_memory_content(memory)
            base_text = f"{index}. {memory_data['text']} (relevance: {score})"

            # Add metadata
            meta_parts = []
            if memory_data.get("category") and memory_data["category"] != "legacy":
                meta_parts.append(f"Category: {memory_data['category']}")
            if memory_data.get("tags"):
                meta_parts.append(f"Tags: {', '.join(memory_data['tags'])}")

            if meta_parts:
                base_text += f" [{'; '.join(meta_parts)}]"

            content_list.append(base_text)

        await emitter.emit(
            description=f"Found {len(search_results)} matching memories",
            status="search_complete",
            done=True,
        )

        return f"Search results for '{query}': {content_list}"

    # Keep existing delete_memory and update_memory methods, but enhance them to handle categorized memories
    async def delete_memory(
        self,
        indices: List[int],
        __user__: dict = None,
        __event_emitter__: Callable[[dict], Any] = None,
    ) -> str:
        """Enhanced delete_memory that works with categorized memories"""
        # ... (keep existing delete_memory implementation, it will work with the new format)
        emitter = EventEmitter(__event_emitter__)

        if not __user__:
            message = "User ID not provided."
            await emitter.emit(description=message, status="missing_user_id", done=True)
            return json.dumps({"message": message}, ensure_ascii=False)

        user_id = __user__.get("id")
        if not user_id:
            message = "User ID not provided."
            await emitter.emit(description=message, status="missing_user_id", done=True)
            return json.dumps({"message": message}, ensure_ascii=False)

        if isinstance(indices, int):
            indices = [indices]

        await emitter.emit(
            description=f"Deleting {len(indices)} memory entries.",
            status="delete_in_progress",
            done=False,
        )

        user_memories = Memories.get_memories_by_user_id(user_id)
        if not user_memories:
            message = "No memories found to delete."
            await emitter.emit(description=message, status="delete_failed", done=True)
            return json.dumps({"message": message}, ensure_ascii=False)

        sorted_memories = sorted(user_memories, key=lambda m: m.created_at)
        responses = []

        for index in indices:
            if index < 1 or index > len(sorted_memories):
                message = f"Memory index {index} does not exist."
                responses.append(message)
                await emitter.emit(
                    description=message, status="delete_failed", done=False
                )
                continue

            memory_to_delete = sorted_memories[index - 1]
            result = Memories.delete_memory_by_id(memory_to_delete.id)
            if not result:
                message = f"Failed to delete memory at index {index}."
                responses.append(message)
                await emitter.emit(
                    description=message, status="delete_failed", done=False
                )
            else:
                message = f"Memory at index {index} deleted successfully."
                responses.append(message)
                await emitter.emit(
                    description=message, status="delete_success", done=False
                )

        await emitter.emit(
            description="All requested memory deletions have been processed.",
            status="delete_complete",
            done=True,
        )
        return json.dumps({"message": "\n".join(responses)}, ensure_ascii=False)

    async def update_memory(
        self,
        updates: List[dict],
        __user__: dict = None,
        __event_emitter__: Callable[[dict], Any] = None,
    ) -> str:
        """Enhanced update_memory that handles categorized memories"""
        # ... (keep existing update_memory implementation, it will work with the new format)
        emitter = EventEmitter(__event_emitter__)

        if not __user__:
            message = "User ID not provided."
            await emitter.emit(description=message, status="missing_user_id", done=True)
            return json.dumps({"message": message}, ensure_ascii=False)

        user_id = __user__.get("id")
        if not user_id:
            message = "User ID not provided."
            await emitter.emit(description=message, status="missing_user_id", done=True)
            return json.dumps({"message": message}, ensure_ascii=False)

        await emitter.emit(
            description=f"Updating {len(updates)} memory entries.",
            status="update_in_progress",
            done=False,
        )

        user_memories = Memories.get_memories_by_user_id(user_id)
        if not user_memories:
            message = "No memories found to update."
            await emitter.emit(description=message, status="update_failed", done=True)
            return json.dumps({"message": message}, ensure_ascii=False)

        sorted_memories = sorted(user_memories, key=lambda m: m.created_at)
        responses = []

        for update_item in updates:
            index = update_item.get("index")
            content = update_item.get("content")

            if index < 1 or index > len(sorted_memories):
                message = f"Memory index {index} does not exist."
                responses.append(message)
                await emitter.emit(
                    description=message, status="update_failed", done=False
                )
                continue

            memory_to_update = sorted_memories[index - 1]
            # Parse existing memory to preserve metadata
            existing_data = self._parse_memory_content(memory_to_update)
            if isinstance(existing_data, dict) and "text" in existing_data:
                # Preserve existing metadata, update text only
                existing_data["text"] = content
                formatted_content = json.dumps(existing_data, ensure_ascii=False)
            else:
                formatted_content = content

            updated_memory = Memories.update_memory_by_id(
                memory_to_update.id, formatted_content
            )
            if not updated_memory:
                message = f"Failed to update memory at index {index}."
                responses.append(message)
                await emitter.emit(
                    description=message, status="update_failed", done=False
                )
            else:
                message = f"Memory at index {index} updated successfully."
                responses.append(message)
                await emitter.emit(
                    description=message, status="update_success", done=False
                )

        await emitter.emit(
            description="All requested memory updates have been processed.",
            status="update_complete",
            done=True,
        )
        return json.dumps({"message": "\n".join(responses)}, ensure_ascii=False)
