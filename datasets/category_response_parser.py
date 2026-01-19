"""
Response Parser for Category-Value Datasets

Parses LLM responses that contain category-value pairs like:
- "visual art: Braque"
- "tools: hook remover"
- "The initial value for visual art was Braque" (RI format)
- "The latest value for tools is screwdriver" (PI format)
"""

import re
from typing import Dict, List, Optional


class CategoryResponseParser:
    """Parse responses containing category-value pairs"""

    @staticmethod
    def parse_response(response: str, category: str) -> Optional[str]:
        """
        Parse a response for a single category query.

        Args:
            response: The LLM response text
            category: The category we're looking for

        Returns:
            Extracted value (or None if not found)
        """
        # Normalize category for matching
        category_norm = category.lower().strip()

        # Pattern 1: "The initial value of category is value" (RI format)
        # Also handles quoted format: "The initial value of category is "value""
        pattern1 = rf"(?:The\s+)?initial\s+value\s+of\s+{re.escape(category)}\s+is\s+[\"']?([A-Za-z0-9\s\-']+?)[\"']?(?:\.|$|\n)"

        # Pattern 1b: "The latest value of category is value" (PI format)
        # Also handles quoted format: "The latest value of category is "value""
        pattern1b = rf"(?:The\s+)?latest\s+value\s+of\s+{re.escape(category)}\s+is\s+[\"']?([A-Za-z0-9\s\-']+?)[\"']?(?:\.|$|\n)"

        # Pattern 2: "category: value" (exact format) - stop at newline, arrow, or punctuation
        # DeepSeek uses format like "category: value → explanation" so we need to stop at →
        # Also handles quoted format: "category: "value""
        pattern2 = rf"{re.escape(category)}\s*:\s*[\"']?([A-Za-z0-9\s\-']+?)[\"']?(?:\n|$|→|\.|\s+-\s+|\s+–\s+)"

        # Pattern 3: "The initial value for category was value" (RI format)
        # Also handles quoted format: "The initial value for category was "value""
        pattern3 = rf"initial\s+value\s+for\s+{re.escape(category)}\s+(?:was|is)\s+[\"']?([A-Za-z0-9\s\-']+?)[\"']?(?:\n|$|\.)"

        # Pattern 3b: "The latest value for category was/is value" (PI format)
        # Also handles quoted format: "The latest value for category was "value""
        pattern3b = rf"latest\s+value\s+for\s+{re.escape(category)}\s+(?:was|is)\s+[\"']?([A-Za-z0-9\s\-']+?)[\"']?(?:\n|$|\.)"

        # Pattern 4: "The INITIAL value for category: value"
        # Also handles quoted format
        pattern4 = rf"INITIAL\s+value\s+for\s+{re.escape(category)}\s*:\s*[\"']?([A-Za-z0-9\s\-']+?)[\"']?(?:\n|$)"

        # Pattern 5: "category was initially value"
        # Also handles quoted format
        pattern5 = rf"{re.escape(category)}\s+was\s+initially\s+[\"']?([A-Za-z0-9\s\-']+?)[\"']?(?:\n|$)"

        # Pattern 6: Direct value mention (more permissive) - stop at newline, arrow, or punctuation
        # "visual art: Braque" or "visual art is Braque" or "visual art was Braque"
        # Also handles quoted format: "visual art: "Braque""
        pattern6 = rf"{re.escape(category)}\s*(?:is|was|:)\s+[\"']?([A-Za-z0-9\s\-']+?)[\"']?(?:\n|$|→|\.|\s+-\s+|\s+–\s+)"

        # Pattern 7: Quoted format: 'category: First appears as "category: value"'
        pattern7 = rf"{re.escape(category)}\s*:\s*First appears as\s+[\"']{re.escape(category)}\s*:\s*([A-Za-z0-9\s\-']+)[\"']"

        # Pattern 8: Format with parenthetical: "category: value (first appearance)"
        pattern8 = rf"{re.escape(category)}\s*:\s*([A-Za-z0-9\s\-']+)\s*\([^)]*(?:first|initial|baseline)[^)]*\)"

        # Pattern 9: Quoted format: "category: value" (DeepSeek Level 50 style)
        pattern9 = rf'"{re.escape(category)}\s*:\s*([A-Za-z0-9\s\-\']+?)"'

        # Pattern 10: Bold markdown format: **category: value** (DeepSeek Level 100 style)
        pattern10 = rf'\*\*{re.escape(category)}\s*:\s*([A-Za-z0-9\s\-\']+?)\*\*'

        # Pattern 11: Numbered list with quotes: "1. category: "value"" (Claude 3.5 Haiku style)
        # Matches: "1. visual art: "video art"" or "23. coffee variety: "natural""
        pattern11 = rf'\d+\.\s+{re.escape(category)}\s*:\s*["\']([A-Za-z0-9\s\-\']+?)["\']'

        # Pattern 12: "The most recent value of category is value" (PI alternate format)
        pattern12 = rf"(?:The\s+)?(?:most\s+recent|last|final)\s+value\s+of\s+{re.escape(category)}\s+is\s+[\"']?([A-Za-z0-9\s\-']+?)[\"']?(?:\.|$|\n)"

        # Try all patterns (prioritize specific patterns over generic ones)
        for pattern in [pattern1, pattern1b, pattern3, pattern3b, pattern12, pattern2, pattern4, pattern5, pattern6, pattern7, pattern8, pattern9, pattern10, pattern11]:
            matches = re.finditer(pattern, response, re.IGNORECASE)

            for match in matches:
                value = match.group(1).strip()

                # Clean up the value
                # Remove trailing punctuation and extra whitespace (including single quotes from reasoning)
                value = re.sub(r'[.,;!?\']$', '', value)
                value = value.strip()

                # Remove common false endings
                value = re.sub(r'\s+(was|is|appears?|first|initial|baseline).*$', '', value, flags=re.IGNORECASE)

                # Remove any newlines that snuck through
                value = value.split('\n')[0].strip()

                # Prefer placeholder format (e.g., "Artist123") over common names (e.g., "Picasso")
                # If we found a common name, continue searching for a placeholder
                if value and not CategoryResponseParser._is_placeholder_format(value):
                    # Store as backup but keep searching
                    if 'backup_value' not in locals():
                        backup_value = value
                    continue

                if value:
                    return value

        # Return backup if no placeholder found
        if 'backup_value' in locals():
            return backup_value

        return None

    @staticmethod
    def _is_placeholder_format(value: str) -> bool:
        """
        Check if value is in placeholder format (e.g., Artist123, Tool50).
        Placeholder format: Capital letter followed by lowercase letters and numbers.
        """
        if not value:
            return False

        # Check for pattern: StartCase followed by digits (e.g., Artist123, Tool50)
        import re
        return bool(re.match(r'^[A-Z][a-z]+[0-9]+$', value))

    @staticmethod
    def parse_batch_response(response: str, expected_categories: List[str]) -> Dict[str, Optional[str]]:
        """
        Parse a response containing values for multiple categories.

        Supports multiple formats:
        1. JSON format: {"category": "value", ...}
        2. Multiple JSON objects (Llama): {"key": "category", "value": "val"} {"key": "category2", "value": "val2"}
        3. Natural language format:
           - RI: "The initial value of category is value..."
           - PI: "The latest value of category is value..."

        Args:
            response: The LLM response text
            expected_categories: List of categories we expect to find

        Returns:
            Dict mapping category -> extracted value (or None if not found)
        """
        results = {}

        # FORMAT 1: Standard JSON object - Try parsing as JSON first (preferred format)
        # IMPORTANT: Handle duplicate keys by taking FIRST non-empty occurrence
        # IMPORTANT: Handle truncated JSON by extracting key-value pairs directly from response
        import json
        try:
            # Check if response contains JSON-like content
            if '"' in response and ':' in response:
                # Check if this is Llama format first (has "key" and "value" fields)
                is_llama_format = '"key"' in response and '"value"' in response

                if not is_llama_format:
                    # Standard JSON format: {"category": "value", ...}
                    # Extract key-value pairs DIRECTLY from response (works even if JSON is truncated)
                    # This avoids the issue where regex fails to match incomplete JSON structure
                    seen_categories = set()

                    # Extract all "key": "value" pairs in order of appearance from ENTIRE response
                    # This regex captures: "category_name": "value_text"
                    # Works even if closing brace } is missing due to truncation
                    key_value_pairs = re.findall(r'"([^"]+)"\s*:\s*"([^"]*)"', response)

                    for key, value in key_value_pairs:
                        key_lower = key.lower()

                        # Find matching expected category (case-insensitive)
                        matched_category = None
                        for expected_cat in expected_categories:
                            if expected_cat.lower() == key_lower:
                                matched_category = expected_cat
                                break

                        # Only take FIRST occurrence with non-empty value
                        if matched_category and matched_category not in seen_categories:
                            value = value.strip()

                            # If value is non-empty and valid, use it
                            if value and CategoryResponseParser.validate_value_format(value):
                                results[matched_category] = value
                                seen_categories.add(matched_category)
                            # If value is empty, skip it (don't mark as seen, allow later valid value)

                    # Fill in missing categories
                    for category in expected_categories:
                        if category not in results:
                            results[category] = None

                    # If we got any valid results, return them
                    if any(v is not None for v in results.values()):
                        return results
        except Exception:
            # Parsing failed, continue to other formats
            pass

        # FORMAT 2: Multiple JSON objects (Llama format)
        # Example: {"key": "fruit variety", "value": "mangosteen"} {"key": "visual art", "value": "art nouveau"}
        try:
            json_objects = re.findall(r'\{\s*"key"\s*:\s*"([^"]+)"\s*,\s*"value"\s*:\s*"([^"]+)"\s*\}', response)
            if json_objects:
                seen_categories = set()
                for key, value in json_objects:
                    # Case-insensitive matching to find corresponding expected category
                    key_lower = key.lower()
                    matched_category = None

                    for expected_cat in expected_categories:
                        if expected_cat.lower() == key_lower:
                            matched_category = expected_cat
                            break

                    # Only take FIRST occurrence of each category (handles duplicates/hallucinations)
                    if matched_category and matched_category not in seen_categories:
                        value = value.strip()
                        if CategoryResponseParser.validate_value_format(value):
                            results[matched_category] = value
                        else:
                            results[matched_category] = None
                        seen_categories.add(matched_category)

                # Fill in missing categories with None
                for category in expected_categories:
                    if category not in results:
                        results[category] = None

                # If we got any valid results, return them
                if any(v is not None for v in results.values()):
                    return results
        except Exception:
            # Regex parsing failed, continue to fallback
            pass

        # FORMAT 3: Fallback - parse using regex patterns (for natural language responses)
        for category in expected_categories:
            value = CategoryResponseParser.parse_response(response, category)
            results[category] = value

        return results

    @staticmethod
    def validate_value_format(value: str) -> bool:
        """
        Validate that a value matches expected format.

        Valid formats:
        - Single words: "Braque", "moraine"
        - Multi-word: "hook remover", "Art Deco"
        - Numbered placeholders: "Artist139", "Tool50"
        """
        if not value:
            return False

        # Check for reasonable length (1-50 characters)
        if len(value) < 1 or len(value) > 50:
            return False

        # Check for valid characters (alphanumeric, spaces, hyphens, apostrophes)
        if not re.match(r'^[A-Za-z0-9\s\-\']+$', value):
            return False

        return True
