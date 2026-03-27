import json
import math
import re

import requests


def build_prompt(cluster_batch: dict[int, list[str]]) -> str:
    """
    Build a prompt for labeling a batch of clusters.

    Each cluster is represented as:
        {cluster_id: [keyword1, keyword2, ...]}

    The prompt instructs the model to return a JSON mapping
    cluster IDs to short, human-readable labels.

    Args:
        cluster_batch: Dictionary mapping cluster IDs to keyword lists.

    Returns:
        str: Formatted prompt string.
    """
    # Format cluster data into text lines.
    lines = []
    for cluster_id, keywords in cluster_batch.items():
        keywords_text = ", ".join(keywords)
        lines.append(f"{cluster_id}: {keywords_text}")

    clusters_text = "\n".join(lines)

    # Construct prompt with strict output instructions.
    prompt = f"""
        You are labeling clusters of scientific papers.

        Return ONLY a valid JSON object.
        Do not include markdown.
        Do not include explanations.
        Each key must be the cluster id as a string.
        Each value must be a short human-readable research topic label, 2 to 5 words.

        Example format:
        {{
        "3": "Sentence Embedding Models",
        "59": "3D Vision and Depth"
        }}

        Clusters:
        {clusters_text}
        """

    return prompt.strip()


def extract_json(text: str) -> dict[str, str]:
    """
    Extract a JSON object from model output.

    The function first attempts direct parsing. If that fails,
    it searches for a JSON-like substring and parses that.

    Args:
        text: Raw response text from the model.

    Returns:
        dict: Parsed JSON dictionary.

    Raises:
        ValueError: If no valid JSON object can be extracted.
    """
    text = text.strip()

    # Attempt direct JSON parsing.
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
    except json.JSONDecodeError:
        pass

    # Attempt to extract JSON substring using regex.
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        json_text = match.group(0)
        try:
            data = json.loads(json_text)
            if isinstance(data, dict):
                return {str(k): str(v) for k, v in data.items()}
        except json.JSONDecodeError:
            pass

    raise ValueError("Could not parse JSON from Ollama response.")


def generate_labels_with_ollama(
    clusters_keywords: dict[int, list[str]],
    batch_size: int = 10,
    model: str = "llama3.2",
    host: str = "http://localhost:11434",
    timeout: int = 300,
    verbose: bool = True,
) -> dict[int, str]:
    """
    Generate labels for clusters using the Ollama API in batches.

    This function:
    - splits clusters into batches
    - sends prompts to the Ollama model
    - parses JSON responses
    - handles errors gracefully
    - returns a mapping of cluster_id -> label

    Args:
        clusters_keywords: Mapping cluster_id -> list of keywords.
        batch_size: Number of clusters per API request.
        model: Ollama model name.
        host: Ollama server URL.
        timeout: Request timeout in seconds.
        verbose: Whether to print progress messages.

    Returns:
        dict: Mapping of cluster_id to generated label.
    """
    all_labels = {}
    cluster_items = list(clusters_keywords.items())

    # Process clusters in batches.
    for batch_start in range(0, len(cluster_items), batch_size):
        batch_items = cluster_items[batch_start : batch_start + batch_size]
        batch_dict = dict(batch_items)

        # Progress logging.
        if verbose:
            batch_num = batch_start // batch_size + 1
            total_batches = math.ceil(len(cluster_items) / batch_size)
            print(f"Generating labels: batch {batch_num}/{total_batches}")

        # Build prompt for current batch.
        prompt = build_prompt(batch_dict)

        try:
            # Send request to Ollama API.
            response = requests.post(
                f"{host}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "format": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                    },
                    "options": {"temperature": 0.1},
                },
                timeout=timeout,
            )
            response.raise_for_status()

            # Extract and parse model output.
            raw_text = response.json()["response"]
            parsed = extract_json(raw_text)

        except requests.exceptions.ConnectionError:
            print(
                f"[ERROR] Could not connect to Ollama at {host}. "
                "Make sure the service is running."
            )
            parsed = {}

        except requests.exceptions.Timeout:
            print(f"[ERROR] Request timed out after {timeout} seconds.")
            parsed = {}

        except requests.exceptions.HTTPError as e:
            print(f"[ERROR] HTTP error from Ollama: {str(e)}")
            parsed = {}

        except ValueError:
            print("[ERROR] Invalid JSON returned by Ollama.")
            parsed = {}

        except Exception as e:
            print(f"[ERROR] Unexpected error: {str(e)}")
            parsed = {}

        # Assign labels for current batch.
        for cluster_id in batch_dict:
            label = (parsed.get(str(cluster_id)) or "").strip()
            all_labels[cluster_id] = label

    return all_labels
