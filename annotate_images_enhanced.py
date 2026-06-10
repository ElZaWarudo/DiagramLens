#!/usr/bin/env python3
"""
annotate_images.py - Enhanced with context-aware categorization

Read a markdown file, find all image references, ask a multimodal model
to (1) categorize the diagram type using surrounding context and (2) generate a detailed technical description.
Two markdown files are written:

* <output_md> – the original markdown with technical descriptions inserted after each image.
* <summary_md> – a structured list of all diagrams with their categories and descriptions.

Usage
-----
    uv run img-parse/annotate_images.py \
        --input  path/to/file.md \
        --output path/to/file_annotated.md \
        --summary path/to/summary.md \
        --categories img-parse/image_categories.json \
        --model qwen3-vl:30b   # any supported vision model

Dependencies (install with uv)
---------------
    uv add requests pillow rich python-dotenv
"""

import argparse
import json
import os
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

from rich.console import Console
from rich.progress import track
from vision_clients import create_vision_client, VisionClient
from PIL import Image

MAX_IMAGE_SIZE = 5 * 1024 * 1024  # 5 MiB
CONTEXT_CHARS = 500  # Characters of context to extract before/after image


# ---------------------------------------------------------------
# Markdown processing
# ---------------------------------------------------------------
IMG_REGEX = re.compile(r"!\[(?P<alt>.*?)\]\((?P<path>[^)]+)\)")


def find_image_refs_with_context(md_text: str, context_size: int = CONTEXT_CHARS) -> List[Dict[str, Any]]:
    """
    Find all markdown image references in the text with surrounding context.
    Returns: List of dictionaries containing image info and context
    """
    matches = []
    for m in IMG_REGEX.finditer(md_text):
        # Extract surrounding context
        start_idx = m.start()
        end_idx = m.end()
        
        # Get context before (look for paragraph or section breaks)
        context_start = max(0, start_idx - context_size)
        text_before = md_text[context_start:start_idx]
        
        # Try to find the start of the paragraph/section
        para_breaks = [text_before.rfind('\n\n'), text_before.rfind('\n#')]
        para_start = max(para_breaks)
        if para_start > 0:
            text_before = text_before[para_start:].strip()
        
        # Get context after
        context_end = min(len(md_text), end_idx + context_size)
        text_after = md_text[end_idx:context_end]
        
        # Try to find the end of the paragraph/section
        para_breaks = [text_after.find('\n\n'), text_after.find('\n#')]
        para_end = min([p for p in para_breaks if p > 0], default=len(text_after))
        text_after = text_after[:para_end].strip()
        
        # Extract any headings above the image
        heading_search = md_text[max(0, start_idx - 1000):start_idx]
        heading_match = re.findall(r'^#+\s+(.+)$', heading_search, re.MULTILINE)
        current_heading = heading_match[-1] if heading_match else ""
        
        matches.append({
            "path": m.group("path"),
            "alt_text": m.group("alt"),
            "start": start_idx,
            "end": end_idx,
            "text_before": text_before,
            "text_after": text_after,
            "current_heading": current_heading,
            "full_match": m.group(0)
        })
    return matches


def pre_categorize_with_context(
    context_info: Dict[str, str],
    categories: List[str],
    provider: str,
    model: str,
    vision_client: VisionClient,
    temperature: float = 0.1
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """
    Use surrounding text context to predict the diagram type before image analysis.
    Returns a predicted category or None if uncertain.
    """
    prompt = f"""Based on the surrounding text context, predict what type of diagram is being referenced.

Current section heading: {context_info['current_heading'] or 'None'}
Image alt text: {context_info['alt_text'] or 'None'}

Text BEFORE the image:
{context_info['text_before'][:300] if context_info['text_before'] else 'None'}

Text AFTER the image:
{context_info['text_after'][:300] if context_info['text_after'] else 'None'}

Based on this context, what type of diagram is most likely being shown?
Available categories: {', '.join(categories)}

Look for keywords that indicate the diagram type:
- "use case", "actors", "system boundary" → use case diagram
- "C4", "context", "container", "component level" → C4 Model diagram
- "entity", "relationship", "ERD", "database model" → entity relationship diagram
- "class", "inheritance", "UML" → class diagram
- "sequence", "message", "lifeline" → sequence diagram
- "state machine", "transitions", "states" → state diagram
- "deployment", "nodes", "infrastructure" → deployment diagram
- "flow", "process", "data flow" → data flow diagram or flowchart
- "architecture", "system design", "components" → architecture diagram
- "mockup", "wireframe", "UI", "interface" → UI/UX design
- "network", "topology", "connectivity" → network diagram
- "git", "branch", "merge" → git workflow

Reply with ONLY the most likely category name. If you cannot determine with reasonable confidence, reply with "unknown"."""

    try:
        response, telemetry = invoke_vision(
            vision_client=vision_client,
            provider=provider,
            model=model,
            prompt=prompt,
            temperature=temperature,
            phase="pre_categorization",
            image_ref=context_info["path"],
        )
    except Exception as exc:
        sys.stderr.write(f"[ERROR] Vision request failed during pre-categorization: {exc}\n")
        return None, None
    
    # Normalize the response
    if response:
        response = response.lower().strip()
        if response in [c.lower() for c in categories]:
            return response, telemetry
    return None, telemetry


def load_categories_config(json_path: Path) -> Dict[str, Any]:
    """Load categories and their technical description prompts."""
    with json_path.open(encoding="utf-8") as f:
        return json.load(f)


def build_fallback_description_prompt(category: str) -> str:
    return f"""Describe this {category} briefly but concretely.

List only:
1. Main classes or elements
2. Key relationships and cardinalities if visible
3. Inheritance, aggregation, or composition if visible
4. Any obvious notation or quality issues

If something is unclear, say so instead of guessing."""


def coerce_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def coerce_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def estimate_tokens_from_text(text: str) -> int:
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def estimate_cost_usd(
    provider: str,
    model: str,
    tokens_input: int,
    tokens_output: int,
    cache_read_input_tokens: int = 0,
    cache_creation_input_tokens: int = 0,
    provider_reported_cost_usd: float | None = None,
) -> float:
    if provider_reported_cost_usd is not None and provider_reported_cost_usd > 0:
        return round(provider_reported_cost_usd, 6)

    provider_key = provider.strip().casefold()
    if provider_key == "ollama":
        return 0.0

    if provider_key in {"opencode-go", "opencode_go", "opencode"}:
        model_key = model.strip().casefold()
        pricing_table = {
            "qwen3.7-max": [
                {"max_total_input_tokens": None, "input": 2.50, "output": 7.50, "cache_read": 0.50, "cache_write": 3.125},
            ],
            "qwen3.7-plus": [
                {"max_total_input_tokens": 256000, "input": 0.40, "output": 1.60, "cache_read": 0.04, "cache_write": 0.50},
                {"max_total_input_tokens": None, "input": 1.20, "output": 4.80, "cache_read": 0.12, "cache_write": 1.50},
            ],
            "qwen3.6-plus": [
                {"max_total_input_tokens": 256000, "input": 0.50, "output": 3.00, "cache_read": 0.05, "cache_write": 0.625},
                {"max_total_input_tokens": None, "input": 2.00, "output": 6.00, "cache_read": 0.20, "cache_write": 2.50},
            ],
        }
        tiers = pricing_table.get(model_key)
        if not tiers:
            return 0.0
        total_input_for_tier = tokens_input + cache_read_input_tokens + cache_creation_input_tokens
        tier = tiers[-1]
        for candidate in tiers:
            max_tokens = candidate["max_total_input_tokens"]
            if max_tokens is None or total_input_for_tier <= max_tokens:
                tier = candidate
                break
        return round(
            (tokens_input / 1_000_000) * tier["input"]
            + (tokens_output / 1_000_000) * tier["output"]
            + (cache_read_input_tokens / 1_000_000) * tier["cache_read"]
            + (cache_creation_input_tokens / 1_000_000) * tier["cache_write"],
            6,
        )

    return 0.0


def invoke_vision(
    vision_client: VisionClient,
    *,
    provider: str,
    model: str,
    prompt: str,
    image_path: Optional[Path] = None,
    temperature: float = 0.0,
    phase: str,
    image_ref: str,
) -> Tuple[str, Dict[str, Any]]:
    started = time.perf_counter()
    response = vision_client.generate(
        model=model,
        prompt=prompt,
        image_path=image_path,
        temperature=temperature,
    )
    latency_seconds = round(time.perf_counter() - started, 4)
    meta = getattr(vision_client, "last_response_meta", None) or {}
    usage = meta.get("usage") if isinstance(meta, dict) and isinstance(meta.get("usage"), dict) else {}
    input_tokens = coerce_int(usage.get("input_tokens")) or estimate_tokens_from_text(prompt)
    output_tokens = coerce_int(usage.get("output_tokens")) or estimate_tokens_from_text(response)
    cache_read_input_tokens = coerce_int(usage.get("cache_read_input_tokens"))
    cache_creation_input_tokens = coerce_int(usage.get("cache_creation_input_tokens"))
    provider_reported_cost = coerce_float(meta.get("cost")) if isinstance(meta, dict) else None
    telemetry = {
        "phase": phase,
        "image_ref": image_ref,
        "provider": provider,
        "model": model,
        "tokens_input": input_tokens,
        "tokens_output": output_tokens,
        "tokens_total": input_tokens + output_tokens,
        "cache_read_input_tokens": cache_read_input_tokens,
        "cache_creation_input_tokens": cache_creation_input_tokens,
        "latency_seconds": latency_seconds,
        "provider_reported_cost_usd": provider_reported_cost,
        "cost_estimated_usd": estimate_cost_usd(
            provider,
            model,
            input_tokens,
            output_tokens,
            cache_read_input_tokens=cache_read_input_tokens,
            cache_creation_input_tokens=cache_creation_input_tokens,
            provider_reported_cost_usd=provider_reported_cost,
        ),
        "usage_source": "provider_usage" if usage else "estimated_chars_div_4",
    }
    return response, telemetry


def build_telemetry_summary(
    *,
    provider: str,
    model: str,
    input_md_path: Path,
    output_md_path: Path,
    summary_md_path: Path,
    calls: List[Dict[str, Any]],
    diagrams_total: int,
) -> Dict[str, Any]:
    return {
        "provider": provider,
        "model": model,
        "input_path": str(input_md_path),
        "output_path": str(output_md_path),
        "summary_path": str(summary_md_path),
        "diagrams_total": diagrams_total,
        "model_calls_total": len(calls),
        "tokens_input": sum(call["tokens_input"] for call in calls),
        "tokens_output": sum(call["tokens_output"] for call in calls),
        "tokens_total": sum(call["tokens_total"] for call in calls),
        "cache_read_input_tokens": sum(call.get("cache_read_input_tokens", 0) for call in calls),
        "cache_creation_input_tokens": sum(call.get("cache_creation_input_tokens", 0) for call in calls),
        "latency_seconds": round(sum(call["latency_seconds"] for call in calls), 4),
        "cost_estimated_usd": round(sum(call.get("cost_estimated_usd", 0.0) for call in calls), 6),
        "calls": calls,
    }


# ---------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate technical descriptions of diagrams in markdown files."
    )
    parser.add_argument("--input", required=True, help="Path to source .md file")
    parser.add_argument("--output", required=True, help="Path for annotated .md file")
    parser.add_argument("--summary", help="Path for summary .md file")
    parser.add_argument("--telemetry-output", help="Path for operational telemetry JSON output")
    parser.add_argument(
        "--categories",
        default=str(Path(__file__).resolve().parent / "image_categories_enhanced.json"),
        help="JSON file with diagram categories and prompts",
    )
    parser.add_argument(
        "--provider",
        default="ollama",
        help="Vision provider to use: ollama or opencode-go (default: ollama)",
    )
    parser.add_argument(
        "--model",
        default="qwen3-vl:30b",
        help="Vision model identifier for the selected provider",
    )
    parser.add_argument(
        "--api-key",
        help="API key for hosted providers. If omitted, environment variables are used.",
    )
    parser.add_argument(
        "--base-url",
        help="Override provider base URL or endpoint when needed.",
    )
    parser.add_argument(
        "--context-size",
        type=int,
        default=500,
        help="Characters of context to analyze around images (default: 500)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed progress",
    )
    args = parser.parse_args()
    
    console = Console()

    # Resolve paths
    input_md_path = Path(args.input).resolve()
    output_md_path = Path(args.output).resolve()
    summary_md_path = (
        Path(args.summary).resolve()
        if args.summary
        else output_md_path.with_name(f"{output_md_path.stem}_summary.md")
    )
    telemetry_output_path = Path(args.telemetry_output).resolve() if args.telemetry_output else None
    categories_path = Path(args.categories).resolve()
    vision_client = create_vision_client(
        provider=args.provider,
        api_key=args.api_key,
        base_url=args.base_url,
    )

    # Validate input
    if not input_md_path.is_file():
        console.print(f"[red]Error: Input file not found: {input_md_path}[/red]")
        sys.exit(1)

    # Load markdown and find images with context
    if args.verbose:
        console.print(f"[cyan]Reading: {input_md_path}[/cyan]")
    
    md_text = input_md_path.read_text(encoding="utf-8")
    image_refs = find_image_refs_with_context(md_text, args.context_size)
    
    if not image_refs:
        console.print("[yellow]No images found in markdown file.[/yellow]")
        sys.exit(0)
    
    console.print(f"[green]Found {len(image_refs)} diagram(s) to process[/green]")

    # Load configuration
    config = load_categories_config(categories_path)
    categories = config.get("categories", [])
    category_prompts = config.get("category_prompts", {})
    
    if not categories:
        console.print("[red]Error: No categories in configuration file.[/red]")
        sys.exit(1)

    # Process each image
    new_md_parts = []
    summary_lines = [
        "# Diagram Analysis Summary\n",
        f"**Source Document:** {input_md_path.name}\n",
        f"**Total Diagrams:** {len(image_refs)}\n",
        "\n---\n\n"
    ]
    
    last_idx = 0
    category_counts = {}
    context_predictions = {"correct": 0, "total": 0}
    telemetry_calls: List[Dict[str, Any]] = []

    # Progress tracking
    iterator = track(image_refs, description="Processing diagrams...") if not args.verbose else image_refs

    for idx, img_info in enumerate(iterator, 1):
        # Preserve markdown up to image
        new_md_parts.append(md_text[last_idx:img_info['start']])
        new_md_parts.append(img_info['full_match'])

        # Normalize path for Unicode issues
        img_path_str_norm = unicodedata.normalize('NFC', img_info['path']).strip()
        img_path = (input_md_path.parent / img_path_str_norm).resolve()
        
        if args.verbose:
            console.print(f"\n[cyan]Processing [{idx}/{len(image_refs)}]: {img_path.name}[/cyan]")
            if img_info['current_heading']:
                console.print(f"  [dim]Section: {img_info['current_heading']}[/dim]")
        
        # Default values
        category = "unknown"
        predicted_category = None
        description = ""
        
        # Check if file exists and is valid
        if not img_path.is_file():
            description = f"⚠️ Image file not found: `{img_info['path']}`"
            console.print(f"[red]Missing: {img_path}[/red]")
        else:
            try:
                Image.open(img_path).verify()
            except Exception as e:
                description = f"⚠️ Invalid image file: `{img_info['path']}`"
                console.print(f"[red]Invalid image: {e}[/red]")
            else:
                # Check size limit
                if img_path.stat().st_size > MAX_IMAGE_SIZE:
                    description = f"⚠️ Image too large (>{MAX_IMAGE_SIZE//1024//1024} MB)"
                    console.print(f"[yellow]Skipping large file[/yellow]")
                else:
                    # Step 1: Pre-categorize using context
                    if args.verbose:
                        console.print("  [dim]Analyzing context for category hints...[/dim]")
                    
                    predicted_category, context_telemetry = pre_categorize_with_context(
                        img_info,
                        categories,
                        args.provider,
                        args.model,
                        vision_client,
                        temperature=0.1
                    )
                    if context_telemetry:
                        telemetry_calls.append(context_telemetry)
                    
                    if predicted_category and args.verbose:
                        console.print(f"  [blue]Context suggests: {predicted_category}[/blue]")
                    
                    # Step 2: Categorize the diagram with context hint
                    cat_prompt = f"""Identify the type of this software engineering diagram.

{f"Context suggests this might be a {predicted_category} diagram." if predicted_category and predicted_category != "unknown" else ""}
{f"Section heading: {img_info['current_heading']}" if img_info['current_heading'] else ""}

Examine the visual elements carefully and choose ONE category from: {', '.join(categories)}

Key distinguishing features:
- C4 Model: Has explicit C4 level labels (Context/Container/Component), technology tags in brackets
- Use Case: Has actors (stick figures), ovals for use cases, system boundary rectangle
- Class Diagram: Shows classes with attributes/methods, inheritance arrows
- Entity Relationship: Shows entities with attributes, relationship lines with cardinality
- Architecture: Shows system components, layers, external services
- Sequence: Has lifelines, messages between objects, activation boxes
- Data Flow: Has numbered processes, data stores, external entities

Reply with only the category name, nothing else."""
                    
                    if args.verbose:
                        console.print("  [dim]Detecting diagram type from image...[/dim]")
                    
                    try:
                        category_response, category_telemetry = invoke_vision(
                            vision_client=vision_client,
                            provider=args.provider,
                            model=args.model,
                            prompt=cat_prompt,
                            image_path=img_path,
                            temperature=0.0,
                            phase="diagram_categorization",
                            image_ref=img_info["path"],
                        )
                        telemetry_calls.append(category_telemetry)
                    except Exception as exc:
                        console.print(f"[red]Category request failed: {exc}[/red]")
                        category_response = ""
                    
                    # Normalize category
                    category = category_response.lower().strip()
                    if category not in [c.lower() for c in categories]:
                        category = "other"
                    
                    # Track context prediction accuracy
                    if predicted_category:
                        context_predictions["total"] += 1
                        if predicted_category == category:
                            context_predictions["correct"] += 1
                    
                    # Count categories
                    category_counts[category] = category_counts.get(category, 0) + 1
                    
                    if args.verbose:
                        console.print(f"  [green]Final type: {category}[/green]")
                        if predicted_category and predicted_category != category:
                            console.print(f"  [yellow]Context prediction was different[/yellow]")

                    # Step 3: Generate technical description
                    desc_prompt = category_prompts.get(
                        category,
                        category_prompts.get("other", {})
                    ).get("prompt", "Describe this diagram in detail.")
                    
                    # Add context to description prompt if available
                    if img_info['text_before'] or img_info['text_after']:
                        desc_prompt += f"\n\nAdditional context from the document:\n"
                        if img_info['text_before']:
                            desc_prompt += f"Before image: {img_info['text_before'][:200]}\n"
                        if img_info['text_after']:
                            desc_prompt += f"After image: {img_info['text_after'][:200]}\n"
                    
                    if args.verbose:
                        console.print("  [dim]Generating technical description...[/dim]")
                    
                    try:
                        description, description_telemetry = invoke_vision(
                            vision_client=vision_client,
                            provider=args.provider,
                            model=args.model,
                            prompt=desc_prompt,
                            image_path=img_path,
                            temperature=0.1,
                            phase="diagram_description",
                            image_ref=img_info["path"],
                        )
                        telemetry_calls.append(description_telemetry)
                    except Exception as exc:
                        console.print(f"[red]Description request failed: {exc}[/red]")
                        description = ""

                    if not description:
                        fallback_prompt = build_fallback_description_prompt(category)
                        try:
                            description, fallback_telemetry = invoke_vision(
                                vision_client=vision_client,
                                provider=args.provider,
                                model=args.model,
                                prompt=fallback_prompt,
                                image_path=img_path,
                                temperature=0.1,
                                phase="fallback_description",
                                image_ref=img_info["path"],
                            )
                            telemetry_calls.append(fallback_telemetry)
                            if args.verbose and description:
                                console.print("  [yellow]Used fallback description prompt[/yellow]")
                        except Exception as exc:
                            console.print(f"[red]Fallback description request failed: {exc}[/red]")
                            description = ""
                    
                    if not description:
                        description = "No description generated."
                    
                    if args.verbose and len(description) > 80:
                        console.print(f"  [dim]{description[:80]}...[/dim]")

        # Add technical description to markdown
        desc_block = (
            f"\n\n**Diagram Type:** {category.replace('_', ' ').title()}\n\n"
            f"**Technical Description:**\n{description}\n\n"
        )
        new_md_parts.append(desc_block)

        # Add to summary with context info
        summary_lines.append(f"## Diagram {idx}: {os.path.basename(img_info['path'])}\n\n")
        summary_lines.append(f"![{img_info['alt_text'] or os.path.basename(img_info['path'])}]({img_info['path']})\n\n")
        summary_lines.append(f"- **Type:** {category.replace('_', ' ').title()}\n")
        if predicted_category and predicted_category != category:
            summary_lines.append(f"- **Context Prediction:** {predicted_category.replace('_', ' ').title()} (mismatch)\n")
        summary_lines.append(f"- **File:** `{img_info['path']}`\n")
        if img_info['current_heading']:
            summary_lines.append(f"- **Section:** {img_info['current_heading']}\n")
        summary_lines.append(f"- **Description:**\n\n{description}\n\n")
        summary_lines.append("---\n\n")

        last_idx = img_info['end']

    # Add remaining content
    new_md_parts.append(md_text[last_idx:])

    # Add statistics to summary
    stats_lines = ["## Analysis Statistics\n\n"]
    
    # Category distribution
    if category_counts:
        stats_lines.append("### Category Distribution\n\n")
        for cat, count in sorted(category_counts.items()):
            percentage = (count / len(image_refs)) * 100
            stats_lines.append(f"- **{cat.replace('_', ' ').title()}:** {count} ({percentage:.1f}%)\n")
    
    # Context prediction accuracy
    if context_predictions["total"] > 0:
        accuracy = (context_predictions["correct"] / context_predictions["total"]) * 100
        stats_lines.append(f"\n### Context Prediction Accuracy\n\n")
        stats_lines.append(f"- **Correct predictions:** {context_predictions['correct']}/{context_predictions['total']} ({accuracy:.1f}%)\n")
    
    stats_lines.append("\n---\n\n")
    
    # Insert after the header
    summary_lines[4:4] = stats_lines

    # Write output files
    output_md_path.parent.mkdir(parents=True, exist_ok=True)
    output_md_path.write_text("".join(new_md_parts), encoding="utf-8")
    
    summary_md_path.parent.mkdir(parents=True, exist_ok=True)
    summary_md_path.write_text("".join(summary_lines), encoding="utf-8")

    if telemetry_output_path:
        telemetry_output_path.parent.mkdir(parents=True, exist_ok=True)
        telemetry_output_path.write_text(
            json.dumps(
                build_telemetry_summary(
                    provider=args.provider,
                    model=args.model,
                    input_md_path=input_md_path,
                    output_md_path=output_md_path,
                    summary_md_path=summary_md_path,
                    calls=telemetry_calls,
                    diagrams_total=len(image_refs),
                ),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    # Final report
    console.print(f"\n[green bold]Processing complete[/green bold]")
    console.print(f"Annotated document: [cyan]{output_md_path}[/cyan]")
    console.print(f"Summary document: [cyan]{summary_md_path}[/cyan]")
    if telemetry_output_path:
        console.print(f"Telemetry: [cyan]{telemetry_output_path}[/cyan]")
    
    if category_counts:
        console.print(f"\n[yellow]Category Distribution:[/yellow]")
        for cat, count in sorted(category_counts.items()):
            console.print(f"  - {cat.replace('_', ' ').title()}: {count}")
    
    if context_predictions["total"] > 0:
        accuracy = (context_predictions["correct"] / context_predictions["total"]) * 100
        console.print(f"\n[cyan]Context Prediction Accuracy: {accuracy:.1f}%[/cyan]")


if __name__ == "__main__":
    main()
