"""Road-network download and deterministic fallback construction."""

from __future__ import annotations

import csv
import importlib
import importlib.util
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from data_pipeline.common import haversine_km, write_csv


@dataclass
class RoadGraph:
    """Small serializable directed road graph used by the data pipeline."""

    nodes: dict[str, dict[str, float]]
    edges: list[dict[str, Any]]


def create_fallback_graph(config: dict[str, Any], rows: int = 5, columns: int = 5) -> RoadGraph:
    """Create a deterministic bidirectional grid within the configured bbox."""
    bbox = config["city"]["bbox"]
    nodes: dict[str, dict[str, float]] = {}
    for row in range(rows):
        lat = bbox["south"] + (bbox["north"] - bbox["south"]) * row / (rows - 1)
        for column in range(columns):
            lon = bbox["west"] + (bbox["east"] - bbox["west"]) * column / (columns - 1)
            node_id = f"n{row:02d}_{column:02d}"
            nodes[node_id] = {"node_id": node_id, "x": float(lon), "y": float(lat)}

    edges: list[dict[str, Any]] = []
    speed_kph = float(config.get("truck", {}).get("speed_kmph", 25.0))
    for row in range(rows):
        for column in range(columns):
            source = f"n{row:02d}_{column:02d}"
            for next_row, next_column in ((row + 1, column), (row, column + 1)):
                if next_row >= rows or next_column >= columns:
                    continue
                target = f"n{next_row:02d}_{next_column:02d}"
                length_m = haversine_km(nodes[source]["y"], nodes[source]["x"], nodes[target]["y"], nodes[target]["x"]) * 1000
                for from_node, to_node in ((source, target), (target, source)):
                    edges.append({
                        "from_node": from_node,
                        "to_node": to_node,
                        "length_m": round(length_m, 6),
                        "speed_kph": speed_kph,
                        "travel_time_min": round(length_m / 1000 / speed_kph * 60, 6),
                    })
    return RoadGraph(nodes, edges)


def _from_osmnx(config: dict[str, Any]) -> RoadGraph:
    ox = importlib.import_module("osmnx")
    bbox = config["city"]["bbox"]
    try:
        graph = ox.graph_from_bbox((bbox["west"], bbox["south"], bbox["east"], bbox["north"]), network_type="drive")
    except TypeError:  # osmnx < 2.0 signature
        graph = ox.graph_from_bbox(bbox["north"], bbox["south"], bbox["east"], bbox["west"], network_type="drive")
    graph = ox.add_edge_speeds(graph)
    graph = ox.add_edge_travel_times(graph)
    nodes = {
        str(node): {"node_id": str(node), "x": float(data["x"]), "y": float(data["y"])}
        for node, data in graph.nodes(data=True)
    }
    edges = []
    for source, target, data in graph.edges(data=True):
        speed = float(data.get("speed_kph", 25.0))
        length = float(data.get("length", 0.0))
        travel_seconds = float(data.get("travel_time", length / 1000 / speed * 3600))
        edges.append({"from_node": str(source), "to_node": str(target), "length_m": length,
                      "speed_kph": speed, "travel_time_min": travel_seconds / 60})
    return RoadGraph(nodes, edges)


def build_road_graph(config: dict[str, Any], fallback: bool = False) -> tuple[RoadGraph, list[str]]:
    warnings: list[str] = []
    if fallback:
        return create_fallback_graph(config), warnings
    if importlib.util.find_spec("osmnx") is None:
        warnings.append("osmnx is unavailable; used deterministic fallback road graph")
        return create_fallback_graph(config), warnings
    try:
        return _from_osmnx(config), warnings
    except Exception as exc:  # external download/library failures intentionally degrade
        warnings.append(f"OSM download failed ({type(exc).__name__}: {exc}); used deterministic fallback road graph")
        return create_fallback_graph(config), warnings


def nearest_node(graph: RoadGraph, lat: float, lon: float) -> str:
    return min(graph.nodes, key=lambda node: haversine_km(lat, lon, graph.nodes[node]["y"], graph.nodes[node]["x"]))


def save_road_graph(graph: RoadGraph, output_dir: Path) -> dict[str, Path]:
    nodes_path = output_dir / "road_nodes.csv"
    edges_path = output_dir / "road_edges.csv"
    graphml_path = output_dir / "road_graph.graphml"
    write_csv(nodes_path, graph.nodes.values(), ["node_id", "x", "y"])
    write_csv(edges_path, graph.edges, ["from_node", "to_node", "length_m", "speed_kph", "travel_time_min"])

    root = ET.Element("graphml", xmlns="http://graphml.graphdrawing.org/xmlns")
    for key_id, target, name, value_type in (
        ("x", "node", "x", "double"), ("y", "node", "y", "double"),
        ("length_m", "edge", "length_m", "double"), ("speed_kph", "edge", "speed_kph", "double"),
        ("travel_time_min", "edge", "travel_time_min", "double"),
    ):
        ET.SubElement(root, "key", id=key_id, **{"for": target, "attr.name": name, "attr.type": value_type})
    graph_element = ET.SubElement(root, "graph", edgedefault="directed")
    for node_id, data in graph.nodes.items():
        node_element = ET.SubElement(graph_element, "node", id=node_id)
        ET.SubElement(node_element, "data", key="x").text = str(data["x"])
        ET.SubElement(node_element, "data", key="y").text = str(data["y"])
    for index, edge in enumerate(graph.edges):
        edge_element = ET.SubElement(graph_element, "edge", id=f"e{index}", source=edge["from_node"], target=edge["to_node"])
        for key in ("length_m", "speed_kph", "travel_time_min"):
            ET.SubElement(edge_element, "data", key=key).text = str(edge[key])
    ET.ElementTree(root).write(graphml_path, encoding="utf-8", xml_declaration=True)
    return {"road_graph": graphml_path, "road_nodes": nodes_path, "road_edges": edges_path}
