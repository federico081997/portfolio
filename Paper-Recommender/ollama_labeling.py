import re
import requests
import json
import math


def build_prompt(cluster_batch: dict[int, list[str]]) -> str:
    """
    Build a prompt for a batch of clusters.

    Each cluster is provided as:
    {
        cluster_id: [keyword1, keyword2, ...]
    }
    """
    lines = []
    for cluster_id, keywords in cluster_batch.items():
        keywords_text = ", ".join(keywords)
        lines.append(f"{cluster_id}: {keywords_text}")

    clusters_text = "\n".join(lines)

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
    """Try to extract a JSON object from Ollama output"""
    text = text.strip()

    # First try direct JSON parse
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
    except json.JSONDecodeError:
        pass

    # Try to find first JSON object in the text
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
    Generate labels for all clusters in batches.

    Parameters
    ----------
    clusters_keywords : dict[int, list[str]]
        Mapping cluster_id -> top keywords
    batch_size : int
        Number of clusters per Ollama request
    model : str
        Ollama model name
    host : str
        Ollama host URL
    timeout : int
        Request timeout in seconds
    verbose : bool
        Print progress if True

    Returns
    -------
    dict[int, str]
        Mapping cluster_id -> generated label
    """
    all_labels = {}
    cluster_items = list(clusters_keywords.items())

    for batch_start in range(0, len(cluster_items), batch_size):
        batch_items = cluster_items[batch_start : batch_start + batch_size]
        batch_dict = dict(batch_items)

        # Print progress
        if verbose:
            batch_num = batch_start // batch_size + 1
            total_batches = math.ceil(len(cluster_items) / batch_size)
            print(f"Generating Ollama labels for batch {batch_num}/{total_batches}...")

        # Generate labels for a batch of clusters using Ollama
        prompt = build_prompt(batch_dict)

        # Generate response from Ollama API
        try:
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

            raw_text = response.json()["response"]
            parsed = extract_json(raw_text)

        except requests.exceptions.ConnectionError as e:
            print(
                f"[ERROR] Unable to connect to Ollama at {host}. "
                "Ensure the Ollama service is running."
            )
            parsed = {}

        except requests.exceptions.Timeout:
            print(f"[ERROR] Request to Ollama timed out after {timeout} seconds.")
            parsed = {}

        except requests.exceptions.HTTPError as e:
            print(f"[ERROR] Ollama returned an HTTP error: {str(e)}")
            parsed = {}

        except ValueError:
            print(
                "[ERROR] Failed to parse Ollama response. "
                "The output may not be valid JSON."
            )
            parsed = {}

        except Exception as e:
            print(f"[ERROR] Unexpected error during Ollama labeling: {str(e)}")
            parsed = {}

        # Process labels for this batch
        for cluster_id in batch_dict:
            label = (parsed.get(str(cluster_id)) or "").strip()
            all_labels[cluster_id] = label

    return all_labels
