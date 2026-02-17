"""
Supervised pre-training for the tool-classifier pathway.

Goal: teach the text encoder + fusion bridge + tool head to predict the
correct tool from a text prompt *before* RL training begins.  This gives
the model a warm-start so that the RL loop doesn't waste millions of
steps just learning the tool vocabulary.

Design constraints (per user spec):
  1. Multiple prompt variants per tool with stylistic/grammar diversity.
  2. Cross-entropy loss directly on the tool-head logits.
  3. NO image input feedback and NO cursor output feedback.

Usage:
    python -m cadfire.training.pretrain_tools          # CLI
    from cadfire.training.pretrain_tools import pretrain_tool_classifier
    pretrain_tool_classifier(agent, config, num_epochs=30)
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from cadfire.model.cad_agent import CADAgent
from cadfire.tokenizer.bpe import BPETokenizer
from cadfire.utils.config import load_config, tool_list, tool_to_index

# ── Prompt templates per tool ────────────────────────────────────────
# Each tool gets a list of natural-language prompts the agent might see.
# Templates use light grammar variance for robustness.

_TOOL_PROMPTS: Dict[str, List[str]] = {
    "NOOP": [
        "Do nothing",
        "No operation needed",
        "Skip this step",
        "Stand by",
        "Wait",
        "Idle",
    ],
    "LINE": [
        "Draw a line",
        "Use the line tool",
        "Create a new line segment",
        "Place a line between two points",
        "Start drawing a straight line",
        "Activate the line command",
        "I need to draw a line",
    ],
    "POLYLINE": [
        "Draw a polyline",
        "Use the polyline tool",
        "Create a connected set of line segments",
        "Start a multi-segment line",
        "Begin polyline drawing",
        "Activate the polyline command",
    ],
    "CIRCLE": [
        "Draw a circle",
        "Use the circle tool",
        "Create a new circle",
        "Place a circle with center and radius",
        "Start the circle command",
        "I need to draw a circle",
    ],
    "ARC": [
        "Draw an arc",
        "Use the arc tool",
        "Create a curved arc segment",
        "Place an arc between points",
        "Start the arc command",
        "Activate arc drawing",
    ],
    "RECTANGLE": [
        "Draw a rectangle",
        "Use the rectangle tool",
        "Create a new rectangle",
        "Place a rectangular shape",
        "Start the rectangle command",
        "I want to draw a rectangle",
    ],
    "POLYGON": [
        "Draw a polygon",
        "Use the polygon tool",
        "Create a regular polygon",
        "Place a polygon shape",
        "Start the polygon command",
        "Draw a hexagon",
        "Create an octagon",
    ],
    "ELLIPSE": [
        "Draw an ellipse",
        "Use the ellipse tool",
        "Create an elliptical shape",
        "Place an ellipse",
        "Start the ellipse command",
        "I need an ellipse",
    ],
    "SPLINE": [
        "Draw a spline",
        "Use the spline tool",
        "Create a smooth curve",
        "Place a spline through points",
        "Start the spline command",
    ],
    "POINT": [
        "Place a point",
        "Use the point tool",
        "Create a point entity",
        "Mark a location with a point",
        "Drop a point here",
    ],
    "MOVE": [
        "Move the selection",
        "Use the move tool",
        "Relocate the selected objects",
        "Translate the selection",
        "Shift the shapes to a new position",
        "Drag the selected entities",
    ],
    "COPY": [
        "Copy the selection",
        "Use the copy tool",
        "Duplicate the selected objects",
        "Clone the selection to a new location",
        "Make a copy of the selected items",
    ],
    "ROTATE": [
        "Rotate the selection",
        "Use the rotate tool",
        "Turn the selected objects",
        "Spin the selection by an angle",
        "Apply rotation to the selection",
    ],
    "SCALE": [
        "Scale the selection",
        "Use the scale tool",
        "Resize the selected objects",
        "Make the selection bigger",
        "Make the selection smaller",
        "Change the size of selected shapes",
    ],
    "MIRROR": [
        "Mirror the selection",
        "Use the mirror tool",
        "Reflect the selected objects",
        "Flip the selection across a line",
        "Create a mirrored copy",
    ],
    "OFFSET": [
        "Offset the selection",
        "Use the offset tool",
        "Create a parallel copy at a distance",
        "Offset the selected shape",
    ],
    "TRIM": [
        "Trim the selection",
        "Use the trim tool",
        "Cut away part of the shape",
        "Trim excess geometry",
    ],
    "EXTEND": [
        "Extend the selection",
        "Use the extend tool",
        "Lengthen the shape to a boundary",
        "Extend the line further",
    ],
    "FILLET": [
        "Fillet the corners",
        "Use the fillet tool",
        "Round the edges",
        "Apply a fillet radius",
    ],
    "CHAMFER": [
        "Chamfer the corners",
        "Use the chamfer tool",
        "Bevel the edges",
        "Apply a chamfer distance",
    ],
    "ARRAY_RECT": [
        "Create a rectangular array",
        "Use the rectangular array tool",
        "Duplicate in a grid pattern",
        "Array in rows and columns",
    ],
    "ARRAY_POLAR": [
        "Create a polar array",
        "Use the polar array tool",
        "Duplicate in a circular pattern",
        "Array around a center point",
    ],
    "EXPLODE": [
        "Explode the selection",
        "Use the explode tool",
        "Break the compound entity apart",
        "Decompose the selection into parts",
    ],
    "JOIN": [
        "Join the selection",
        "Use the join tool",
        "Merge the selected entities",
        "Connect the selected segments",
    ],
    "BREAK": [
        "Break the selection",
        "Use the break tool",
        "Split the entity at a point",
        "Break the shape into pieces",
    ],
    "LENGTHEN": [
        "Lengthen the selection",
        "Use the lengthen tool",
        "Extend or shorten the entity",
        "Adjust the length of the line",
    ],
    "HATCH": [
        "Apply hatching",
        "Use the hatch tool",
        "Fill the region with a pattern",
        "Create a hatch fill",
    ],
    "MTEXT": [
        "Place multiline text",
        "Use the multiline text tool",
        "Create a text block",
        "Add paragraph text",
    ],
    "DTEXT": [
        "Place single line text",
        "Use the text tool",
        "Create a text label",
        "Add a text string",
    ],
    "DIM_LINEAR": [
        "Add a linear dimension",
        "Use the linear dimension tool",
        "Measure horizontal or vertical distance",
        "Place a linear measurement",
    ],
    "DIM_ALIGNED": [
        "Add an aligned dimension",
        "Use the aligned dimension tool",
        "Measure the actual distance between two points",
        "Place an aligned measurement",
    ],
    "DIM_ANGULAR": [
        "Add an angular dimension",
        "Use the angular dimension tool",
        "Measure the angle between two lines",
        "Place an angle measurement",
    ],
    "DIM_RADIUS": [
        "Add a radius dimension",
        "Use the radius dimension tool",
        "Measure the radius of a circle",
        "Place a radius measurement",
    ],
    "DIM_DIAMETER": [
        "Add a diameter dimension",
        "Use the diameter dimension tool",
        "Measure the diameter of a circle",
        "Place a diameter measurement",
    ],
    "LAYER_SET": [
        "Set the active layer",
        "Use the layer set command",
        "Switch to a different layer",
        "Change the current layer",
    ],
    "LAYER_OFF": [
        "Turn off a layer",
        "Use the layer off command",
        "Hide a layer",
        "Make a layer invisible",
    ],
    "LAYER_ON": [
        "Turn on all layers",
        "Use the layer on command",
        "Show all layers",
        "Make all layers visible",
    ],
    "LAYER_FREEZE": [
        "Freeze a layer",
        "Use the layer freeze command",
        "Lock a layer from editing",
        "Freeze the specified layer",
    ],
    "LAYER_THAW": [
        "Thaw all layers",
        "Use the layer thaw command",
        "Unfreeze all layers",
        "Unlock all frozen layers",
    ],
    "COLOR_SET": [
        "Set the drawing color",
        "Use the color set command",
        "Change the active color",
        "Switch to a different color",
    ],
    "LINETYPE_SET": [
        "Set the linetype",
        "Use the linetype set command",
        "Change the line style",
        "Switch to a different linetype",
    ],
    "MATCHPROP": [
        "Match properties",
        "Use the match properties tool",
        "Copy formatting from one entity",
        "Apply properties from a source object",
    ],
    "SELECT": [
        "Select an object",
        "Use the select tool",
        "Click to select a shape",
        "Pick an entity",
        "Choose an object",
    ],
    "MULTISELECT": [
        "Select multiple objects",
        "Use the multiselect tool",
        "Select a region of shapes",
        "Pick several entities at once",
    ],
    "DESELECT": [
        "Deselect all",
        "Use the deselect command",
        "Clear the selection",
        "Unselect everything",
    ],
    "ERASE": [
        "Erase the selection",
        "Use the erase tool",
        "Delete the selected objects",
        "Remove the selected entities",
        "Get rid of the selection",
    ],
    "UNDO": [
        "Undo the last action",
        "Use the undo command",
        "Reverse the previous step",
        "Go back one step",
    ],
    "REDO": [
        "Redo the last undone action",
        "Use the redo command",
        "Reapply the previous step",
        "Go forward one step",
    ],
    "ZOOM_IN": [
        "Zoom in",
        "Use the zoom in command",
        "Magnify the view",
        "Get closer to the drawing",
    ],
    "ZOOM_OUT": [
        "Zoom out",
        "Use the zoom out command",
        "Reduce the magnification",
        "See more of the drawing",
    ],
    "ZOOM_EXTENTS": [
        "Zoom to extents",
        "Use the zoom extents command",
        "Fit everything in the view",
        "Show all entities",
    ],
    "PAN": [
        "Pan the view",
        "Use the pan tool",
        "Scroll the viewport",
        "Move the camera",
        "Shift the view",
    ],
    "FIT_VIEW": [
        "Fit the view",
        "Use the fit view command",
        "Auto-zoom to show all objects",
        "Adjust the viewport to fit everything",
    ],
    "CONFIRM": [
        "Confirm the action",
        "Use the confirm command",
        "Finish the current operation",
        "Accept and complete the step",
        "Press enter to confirm",
    ],
    "CANCEL": [
        "Cancel the action",
        "Use the cancel command",
        "Abort the current operation",
        "Discard and stop",
        "Press escape to cancel",
    ],
}


# ── Dataset ──────────────────────────────────────────────────────────

class ToolPromptDataset(Dataset):
    """Flat dataset of (token_ids, tool_index) pairs."""

    def __init__(self, config: Dict[str, Any] | None = None,
                 tokenizer: BPETokenizer | None = None,
                 augment_per_prompt: int = 1):
        self.config = config or load_config()
        self.tokenizer = tokenizer or BPETokenizer(
            vocab_size=self.config["model"]["text_vocab_size"],
            max_len=self.config["model"]["text_max_len"],
        )
        self._tool_idx = tool_to_index()
        self._tools = tool_list()

        # Build flat list of (prompt_text, tool_index)
        self.samples: List[Tuple[str, int]] = []
        for tool_name in self._tools:
            prompts = _TOOL_PROMPTS.get(tool_name, [f"Activate {tool_name.lower().replace('_', ' ')}"])
            idx = self._tool_idx[tool_name]
            for prompt in prompts:
                for _ in range(augment_per_prompt):
                    self.samples.append((prompt, idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        prompt_text, tool_idx = self.samples[idx]
        ids = self.tokenizer.encode_padded(prompt_text)
        return {
            "text_ids": torch.tensor(ids, dtype=torch.long),
            "tool_target": torch.tensor(tool_idx, dtype=torch.long),
        }


# ── Pretraining loop ────────────────────────────────────────────────

def pretrain_tool_classifier(
    agent: CADAgent,
    config: Dict[str, Any] | None = None,
    num_epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 64,
    device: str | None = None,
    verbose: bool = True,
) -> Dict[str, List[float]]:
    """Run supervised pre-training on text→tool mapping.

    Only the text encoder, fusion bridge, and tool head receive
    gradients.  The vision encoder and cursor head are frozen.

    Args:
        agent: The CADAgent model (modified in-place).
        config: Optional config dict (defaults to global config).
        num_epochs: How many passes over the dataset.
        lr: Learning rate for supervised stage.
        batch_size: Mini-batch size.
        device: torch device string.
        verbose: Print per-epoch stats.

    Returns:
        Dict with ``"losses"`` and ``"accuracies"`` lists (one per epoch).
    """
    config = config or load_config()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    agent = agent.to(device)

    # Freeze everything except text encoder + fusion + tool head
    for name, param in agent.named_parameters():
        param.requires_grad = False
    for module in [agent.text, agent.fusion, agent.tool_head]:
        for param in module.parameters():
            param.requires_grad = True

    # Build dataset & dataloader
    dataset = ToolPromptDataset(config=config)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        drop_last=False)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, agent.parameters()), lr=lr
    )
    criterion = nn.CrossEntropyLoss()

    # We need a dummy image and state vector (zeroed out) since the
    # forward pass requires them, but we zero them and don't backprop
    # through the vision encoder.
    m = config["model"]
    H = config["canvas"]["render_height"]
    W = config["canvas"]["render_width"]
    in_ch = agent.in_channels

    history: Dict[str, List[float]] = {"losses": [], "accuracies": []}

    agent.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in loader:
            text_ids = batch["text_ids"].to(device)
            targets = batch["tool_target"].to(device)
            B = text_ids.size(0)

            # Dummy image & state (all zeros – no visual signal)
            dummy_image = torch.zeros(B, H, W, in_ch, device=device)
            dummy_state = torch.zeros(B, m["state_dim"], device=device)

            obs = {
                "image": dummy_image,
                "text_ids": text_ids,
                "state_vec": dummy_state,
            }

            out = agent(obs)
            logits = out["tool_logits"]  # (B, num_tools)

            loss = criterion(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, agent.parameters()), 1.0
            )
            optimizer.step()

            total_loss += loss.item() * B
            preds = logits.argmax(dim=-1)
            correct += (preds == targets).sum().item()
            total += B

        avg_loss = total_loss / max(total, 1)
        accuracy = correct / max(total, 1)
        history["losses"].append(avg_loss)
        history["accuracies"].append(accuracy)

        if verbose:
            print(f"  Pretrain epoch {epoch + 1:>3d}/{num_epochs} | "
                  f"loss {avg_loss:.4f} | acc {accuracy:.3f}")

    # Unfreeze everything for subsequent RL training
    for param in agent.parameters():
        param.requires_grad = True

    return history


# ── CLI entry-point ──────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Pre-train tool classifier from text prompts")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--save", type=str, default=None,
                        help="Path to save checkpoint after pretraining")
    parser.add_argument("--device", type=str, default=None, help="torch device")
    args = parser.parse_args()

    config = load_config()
    agent = CADAgent(config)

    print(f"Pre-training tool classifier ({len(tool_list())} tools)")
    print(f"Dataset size: {len(ToolPromptDataset(config=config))} samples")

    history = pretrain_tool_classifier(
        agent, config,
        num_epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        device=args.device,
    )

    print(f"\nFinal accuracy: {history['accuracies'][-1]:.3f}")
    print(f"Final loss:     {history['losses'][-1]:.4f}")

    if args.save:
        agent.save_checkpoint(args.save, extra_meta={
            "pretrain_epochs": args.epochs,
            "pretrain_final_acc": history["accuracies"][-1],
        })
        print(f"Saved to {args.save}")


if __name__ == "__main__":
    main()
