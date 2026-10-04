#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Build WiVRn GLB controller models and a profile from Steam Frame render-model assets.

The source OBJ/PNG files are read in place; their mesh and texture data is embedded
in the generated GLBs. Reading local SteamVR assets does not grant redistribution
rights, so review the asset terms before sharing generated output. Output is written
only to the requested directory. The implementation uses Python's standard library
and the adjacent ``frame_controller_png.py`` helper when it is available.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    from frame_controller_png import pack_orm
except ImportError:
    pack_orm = None

DEFAULT_SOURCE_ROOT = Path(
    "/opt/steamvr/drivers/frame_controller/resources/rendermodels"
)
DEFAULT_OUTPUT = Path("frame-controller-profile")
PROFILE_ID = "steam-frame"
COMPONENT_IDS = {
    "trigger": "xr-standard-trigger",
    "button_grip": "xr-standard-squeeze",
    "thumbstick": "xr-standard-thumbstick",
    "button_a": "a-button",
    "button_b": "b-button",
    "button_x": "x-button",
    "button_y": "y-button",
    "bumper": "frame-bumper",
    "button_menu": "frame-menu",
    "button_view": "frame-view",
    "button_system": "frame-system",
}
DPAD_IDS = {
    "up": "frame-dpad-up",
    "down": "frame-dpad-down",
    "left": "frame-dpad-left",
    "right": "frame-dpad-right",
}
DPAD_PATHS = {
    "up": "/input/dpad_up/click",
    "down": "/input/dpad_down/click",
    "left": "/input/dpad_left/click",
    "right": "/input/dpad_right/click",
}


def fail(message: str) -> None:
    raise ValueError(message)


def vec3(value: Iterable[float]) -> tuple[float, float, float]:
    result = tuple(float(x) for x in value)
    if len(result) != 3 or not all(math.isfinite(x) for x in result):
        fail("expected a finite 3D vector")
    return result  # type: ignore[return-value]


def vadd(a: Iterable[float], b: Iterable[float]) -> tuple[float, float, float]:
    aa, bb = tuple(a), tuple(b)
    return (aa[0] + bb[0], aa[1] + bb[1], aa[2] + bb[2])


def vsub(a: Iterable[float], b: Iterable[float]) -> tuple[float, float, float]:
    aa, bb = tuple(a), tuple(b)
    return (aa[0] - bb[0], aa[1] - bb[1], aa[2] - bb[2])


def vscale(a: Iterable[float], scale: float) -> tuple[float, float, float]:
    aa = tuple(a)
    return (aa[0] * scale, aa[1] * scale, aa[2] * scale)


def vdot(a: Iterable[float], b: Iterable[float]) -> float:
    aa, bb = tuple(a), tuple(b)
    return aa[0] * bb[0] + aa[1] * bb[1] + aa[2] * bb[2]


def vcross(a: Iterable[float], b: Iterable[float]) -> tuple[float, float, float]:
    aa, bb = tuple(a), tuple(b)
    return (
        aa[1] * bb[2] - aa[2] * bb[1],
        aa[2] * bb[0] - aa[0] * bb[2],
        aa[0] * bb[1] - aa[1] * bb[0],
    )


def vnorm(a: Iterable[float]) -> float:
    return math.sqrt(vdot(a, a))


def vnormalize(a: Iterable[float], fallback=(0.0, 0.0, 1.0)):
    length = vnorm(a)
    if length < 1e-20:
        return fallback
    return vscale(a, 1.0 / length)


# Quaternions use glTF's [x, y, z, w] convention.
Quat = tuple[float, float, float, float]
IDENTITY: Quat = (0.0, 0.0, 0.0, 1.0)


def qnormalize(q: Iterable[float]) -> Quat:
    values = tuple(float(x) for x in q)
    length = math.sqrt(sum(x * x for x in values))
    if length < 1e-20:
        return IDENTITY
    return tuple(x / length for x in values)  # type: ignore[return-value]


def qmul(a: Quat, b: Quat) -> Quat:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def qconjugate(q: Quat) -> Quat:
    return (-q[0], -q[1], -q[2], q[3])


def qaxis(axis: Iterable[float], degrees: float) -> Quat:
    unit = vnormalize(axis)
    half = math.radians(degrees) * 0.5
    sine = math.sin(half)
    return qnormalize((unit[0] * sine, unit[1] * sine, unit[2] * sine, math.cos(half)))


def qrotate(q: Quat, point: Iterable[float]) -> tuple[float, float, float]:
    p = tuple(point)
    vq = (p[0], p[1], p[2], 0.0)
    rotated = qmul(qmul(q, vq), qconjugate(q))
    return (rotated[0], rotated[1], rotated[2])


def qeuler_xyz(degrees: Iterable[float]) -> Quat:
    x, y, z = vec3(degrees)
    qx = qaxis((1.0, 0.0, 0.0), x)
    qy = qaxis((0.0, 1.0, 0.0), y)
    qz = qaxis((0.0, 0.0, 1.0), z)
    # Steam's XYZ fields describe a local basis rotated X, then Y, then Z.
    # In active quaternion products that is qz * qy * qx. On the Frame source,
    # this exactly matches the grip-button rotation axis after transforming +Y.
    return qnormalize(qmul(qmul(qz, qy), qx))


def bounds(points: Iterable[Iterable[float]]):
    values = [tuple(p) for p in points]
    if not values:
        fail("cannot calculate bounds for an empty point list")
    return (
        [min(point[i] for point in values) for i in range(3)],
        [max(point[i] for point in values) for i in range(3)],
    )


@dataclass
class ObjMesh:
    name: str
    positions: list[tuple[float, float, float]]
    normals: list[tuple[float, float, float]]
    texcoords: list[tuple[float, float]]
    indices: list[int]
    triangle_count: int
    raw_vertex_count: int


def parse_obj(path: Path) -> ObjMesh:
    raw_positions: list[tuple[float, float, float]] = []
    raw_normals: list[tuple[float, float, float]] = []
    raw_uvs: list[tuple[float, float]] = []
    positions: list[tuple[float, float, float]] = []
    normals: list[tuple[float, float, float]] = []
    texcoords: list[tuple[float, float]] = []
    indices: list[int] = []
    unique: dict[tuple[int, int, int], int] = {}
    triangle_count = 0

    def index_of(text: str, count: int) -> int:
        number = int(text)
        index = number - 1 if number > 0 else count + number
        if index < 0 or index >= count:
            fail(f"{path}: OBJ index {number} is out of range")
        return index

    with path.open("r", encoding="utf-8", errors="strict") as source:
        for line_number, raw in enumerate(source, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            kind = parts[0]
            if kind == "v":
                if len(parts) < 4:
                    fail(f"{path}:{line_number}: malformed position")
                raw_positions.append(vec3(parts[1:4]))
            elif kind == "vn":
                if len(parts) < 4:
                    fail(f"{path}:{line_number}: malformed normal")
                raw_normals.append(vec3(parts[1:4]))
            elif kind == "vt":
                if len(parts) < 3:
                    fail(f"{path}:{line_number}: malformed texcoord")
                u, v = float(parts[1]), float(parts[2])
                if not math.isfinite(u) or not math.isfinite(v):
                    fail(f"{path}:{line_number}: non-finite texcoord")
                # OBJ's origin is bottom-left; glTF's image origin is top-left.
                raw_uvs.append((u, 1.0 - v))
            elif kind == "f":
                if len(parts) < 4:
                    fail(f"{path}:{line_number}: face has fewer than three vertices")
                face: list[int] = []
                for token in parts[1:]:
                    fields = token.split("/")
                    if len(fields) < 3 or not fields[0] or not fields[1] or not fields[2]:
                        fail(f"{path}:{line_number}: expected v/vt/vn face indices")
                    key = (
                        index_of(fields[0], len(raw_positions)),
                        index_of(fields[1], len(raw_uvs)),
                        index_of(fields[2], len(raw_normals)),
                    )
                    vertex = unique.get(key)
                    if vertex is None:
                        vertex = len(positions)
                        unique[key] = vertex
                        positions.append(raw_positions[key[0]])
                        texcoords.append(raw_uvs[key[1]])
                        normals.append(vnormalize(raw_normals[key[2]]))
                    face.append(vertex)
                # Valve's files are triangles today; fan triangulation preserves any quads/ngons.
                for i in range(1, len(face) - 1):
                    indices.extend((face[0], face[i], face[i + 1]))
                    triangle_count += 1

    if not positions or not indices:
        fail(f"{path}: no drawable triangles")
    if len(normals) != len(positions) or len(texcoords) != len(positions):
        fail(f"{path}: inconsistent OBJ vertex attributes")
    return ObjMesh(path.stem, positions, normals, texcoords, indices, triangle_count, len(raw_positions))


def png_chunks(data: bytes):
    signature = b"\x89PNG\r\n\x1a\n"
    if not data.startswith(signature):
        fail("invalid PNG signature")
    offset = len(signature)
    while offset + 12 <= len(data):
        size = struct.unpack_from(">I", data, offset)[0]
        kind = data[offset + 4 : offset + 8]
        start = offset + 8
        end = start + size
        if end + 4 > len(data):
            fail("truncated PNG chunk")
        payload = data[start:end]
        crc = struct.unpack_from(">I", data, end)[0]
        if zlib.crc32(kind + payload) & 0xFFFFFFFF != crc:
            fail(f"PNG {kind.decode('ascii', 'replace')} CRC mismatch")
        yield kind, payload
        offset = end + 4
        if kind == b"IEND":
            break


def decode_png(data: bytes):
    width = height = bit_depth = color_type = interlace = None
    idat: list[bytes] = []
    for kind, payload in png_chunks(data):
        if kind == b"IHDR":
            width, height, bit_depth, color_type, compression, filtering, interlace = struct.unpack(
                ">IIBBBBB", payload
            )
            if compression != 0 or filtering != 0:
                fail("unsupported PNG compression/filter method")
        elif kind == b"IDAT":
            idat.append(payload)
    if not width or not height or bit_depth != 8 or interlace != 0:
        fail("only non-interlaced 8-bit PNG files are supported")
    channels_by_type = {0: 1, 2: 3, 4: 2, 6: 4}
    channels = channels_by_type.get(color_type)
    if not channels:
        fail(f"unsupported PNG color type {color_type}")
    compressed = zlib.decompress(b"".join(idat))
    stride = width * channels
    if len(compressed) != height * (stride + 1):
        fail("PNG decompressed size does not match IHDR")

    pixels = bytearray(height * stride)
    source_offset = 0
    for y in range(height):
        filter_type = compressed[source_offset]
        source_offset += 1
        row = bytearray(compressed[source_offset : source_offset + stride])
        source_offset += stride
        prior = pixels[(y - 1) * stride : y * stride] if y else bytes(stride)
        for i in range(stride):
            left = row[i - channels] if i >= channels else 0
            above = prior[i]
            upper_left = prior[i - channels] if i >= channels else 0
            if filter_type == 1:
                row[i] = (row[i] + left) & 255
            elif filter_type == 2:
                row[i] = (row[i] + above) & 255
            elif filter_type == 3:
                row[i] = (row[i] + ((left + above) >> 1)) & 255
            elif filter_type == 4:
                p = left + above - upper_left
                pa, pb, pc = abs(p - left), abs(p - above), abs(p - upper_left)
                predictor = left if pa <= pb and pa <= pc else (above if pb <= pc else upper_left)
                row[i] = (row[i] + predictor) & 255
            elif filter_type != 0:
                fail(f"unsupported PNG filter {filter_type}")
        pixels[y * stride : (y + 1) * stride] = row
    return width, height, color_type, channels, bytes(pixels)


def png_chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    )


def encode_png_rgba(width: int, height: int, rgba: bytes) -> bytes:
    expected = width * height * 4
    if len(rgba) != expected:
        fail("RGBA byte count does not match image dimensions")
    stride = width * 4
    filtered = bytearray(height * (stride + 1))
    for row in range(height):
        start = row * (stride + 1)
        filtered[start] = 0
        filtered[start + 1 : start + 1 + stride] = rgba[row * stride : (row + 1) * stride]
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", ihdr)
        + png_chunk(b"IDAT", zlib.compress(filtered, 6))
        + png_chunk(b"IEND", b"")
    )


def pack_metallic_roughness(rough_path: Path, metal_path: Path) -> bytes:
    rough_data = rough_path.read_bytes()
    metal_data = metal_path.read_bytes()
    rw, rh, rt, rc, rough = decode_png(rough_data)
    mw, mh, mt, mc, metal = decode_png(metal_data)
    if (rw, rh, rt, rc) != (mw, mh, mt, mc) or rc != 1:
        fail("roughness and metalness maps must be same-size grayscale PNGs")
    count = rw * rh
    rgba = bytearray(count * 4)
    rgba[1::4] = rough
    rgba[2::4] = metal
    rgba[3::4] = b"\xff" * count
    return encode_png_rgba(rw, rh, bytes(rgba))


def tangent_vectors(mesh: ObjMesh):
    count = len(mesh.positions)
    tan = [[0.0, 0.0, 0.0] for _ in range(count)]
    bitan = [[0.0, 0.0, 0.0] for _ in range(count)]
    for i in range(0, len(mesh.indices), 3):
        i0, i1, i2 = mesh.indices[i : i + 3]
        p0, p1, p2 = mesh.positions[i0], mesh.positions[i1], mesh.positions[i2]
        uv0, uv1, uv2 = mesh.texcoords[i0], mesh.texcoords[i1], mesh.texcoords[i2]
        e1, e2 = vsub(p1, p0), vsub(p2, p0)
        du1, dv1 = uv1[0] - uv0[0], uv1[1] - uv0[1]
        du2, dv2 = uv2[0] - uv0[0], uv2[1] - uv0[1]
        determinant = du1 * dv2 - du2 * dv1
        if abs(determinant) < 1e-20:
            continue
        reciprocal = 1.0 / determinant
        t = vscale(vsub(vscale(e1, dv2), vscale(e2, dv1)), reciprocal)
        b = vscale(vsub(vscale(e2, du1), vscale(e1, du2)), reciprocal)
        for vertex in (i0, i1, i2):
            for axis in range(3):
                tan[vertex][axis] += t[axis]
                bitan[vertex][axis] += b[axis]
    output = []
    for i, normal in enumerate(mesh.normals):
        projected = vsub(tan[i], vscale(normal, vdot(normal, tan[i])))
        if vnorm(projected) < 1e-15:
            axis = (1.0, 0.0, 0.0) if abs(normal[0]) < 0.8 else (0.0, 1.0, 0.0)
            projected = vcross(axis, normal)
        tangent = vnormalize(projected, (1.0, 0.0, 0.0))
        handedness = -1.0 if vdot(vcross(normal, tangent), bitan[i]) < 0 else 1.0
        output.append((tangent[0], tangent[1], tangent[2], handedness))
    return output


class GlbBuilder:
    def __init__(self, hand: str, grip_origin, grip_rotation: Quat, material: dict[str, Any]):
        self.hand = hand
        self.grip_origin = vec3(grip_origin)
        # openxr_grip is a pose in the controller/model frame: it maps grip-local
        # coordinates to model coordinates. GLB geometry is stored in the inverse
        # frame so a WiVRn grip-space entity places it at the original model pose.
        self.grip_rotation = qnormalize(grip_rotation)
        self.grip_inverse = qconjugate(self.grip_rotation)
        self.material_source = material
        self.nodes: list[dict[str, Any]] = []
        self.meshes: list[dict[str, Any]] = []
        self.buffer_views: list[dict[str, Any]] = []
        self.accessors: list[dict[str, Any]] = []
        self.binary = bytearray()
        self.root = self.add_node("FrameController" + hand.title())
        self.roots = [self.root]
        self.response_nodes: set[str] = set()
        self.response_samples: set[str] = set()
        self.component_nodes: dict[str, str] = {}
        self.expected_mesh_stats: dict[str, tuple[int, int]] = {}
        self.material_index = 0

    def point_to_grip(self, point):
        return qrotate(self.grip_inverse, vsub(point, self.grip_origin))

    def vector_to_grip(self, vector):
        return qrotate(self.grip_inverse, vector)

    def direction_to_grip(self, direction):
        return vnormalize(self.vector_to_grip(direction), (0.0, 1.0, 0.0))

    def orientation_to_grip(self, orientation: Quat):
        # motion.rotate_xyz is the local-plane-to-model orientation. Compose it
        # with the model-to-grip inverse pose to get the plane basis in GLB space.
        return qnormalize(qmul(self.grip_inverse, orientation))

    def validate_grip_pose_roundtrip(self, points: Iterable[Iterable[float]]):
        # Match OpenVR's child-pose convention: GLB local points map back to the
        # Steam render-model frame through model_from_grip * p_grip + origin.
        max_error = 0.0
        for point in points:
            source = vec3(point)
            in_grip = self.point_to_grip(source)
            back_in_model = vadd(
                self.grip_origin,
                qrotate(self.grip_rotation, in_grip),
            )
            max_error = max(max_error, vnorm(vsub(back_in_model, source)))
        if max_error > 1e-9:
            fail(f"{self.hand}: grip pose round-trip error {max_error:.3g} m")
        return max_error

    def add_node(self, name: str, parent: int | None = None, translation=None, rotation=None, mesh=None):
        if any(existing.get("name") == name for existing in self.nodes):
            fail(f"duplicate GLB node name {name}")
        node: dict[str, Any] = {"name": name}
        if translation is not None:
            node["translation"] = [float(x) for x in translation]
        if rotation is not None:
            node["rotation"] = [float(x) for x in qnormalize(rotation)]
        if mesh is not None:
            node["mesh"] = mesh
        self.nodes.append(node)
        index = len(self.nodes) - 1
        if parent is not None:
            self.nodes[parent].setdefault("children", []).append(index)
        return index

    def add_binary(self, data: bytes, target: int | None = None):
        while len(self.binary) & 3:
            self.binary.append(0)
        offset = len(self.binary)
        self.binary.extend(data)
        view: dict[str, Any] = {
            "buffer": 0,
            "byteOffset": offset,
            "byteLength": len(data),
        }
        if target is not None:
            view["target"] = target
        self.buffer_views.append(view)
        return len(self.buffer_views) - 1

    def add_accessor(
        self,
        view: int,
        component_type: int,
        count: int,
        kind: str,
        minimum=None,
        maximum=None,
        normalized=False,
    ):
        accessor: dict[str, Any] = {
            "bufferView": view,
            "componentType": component_type,
            "count": count,
            "type": kind,
        }
        if minimum is not None:
            accessor["min"] = [float(x) for x in minimum]
        if maximum is not None:
            accessor["max"] = [float(x) for x in maximum]
        if normalized:
            accessor["normalized"] = True
        self.accessors.append(accessor)
        return len(self.accessors) - 1

    def add_image(self, name: str, data: bytes):
        view = self.add_binary(data)
        return {"name": name, "bufferView": view, "mimeType": "image/png"}

    def add_mesh(self, name: str, source: ObjMesh, pivot=None):
        # Static meshes are stored directly in grip space. Animated meshes are
        # stored relative to their explicit pivot node so rotations stay local.
        if pivot is None:
            transformed = [self.point_to_grip(position) for position in source.positions]
        else:
            pivot = vec3(pivot)
            pivot_grip = self.point_to_grip(pivot)
            transformed = [
                vsub(self.point_to_grip(position), pivot_grip)
                for position in source.positions
            ]
        transformed_normals = [
            vnormalize(qrotate(self.grip_inverse, normal)) for normal in source.normals
        ]
        transformed_mesh = ObjMesh(
            name,
            transformed,
            transformed_normals,
            source.texcoords,
            source.indices,
            source.triangle_count,
            source.raw_vertex_count,
        )
        tangents = tangent_vectors(transformed_mesh)

        pos_flat = [coordinate for point in transformed for coordinate in point]
        normal_flat = [coordinate for normal in transformed_normals for coordinate in normal]
        uv_flat = [coordinate for uv in source.texcoords for coordinate in uv]
        tangent_flat = [coordinate for tangent in tangents for coordinate in tangent]
        minimum, maximum = bounds(transformed)

        position_view = self.add_binary(struct.pack("<" + "f" * len(pos_flat), *pos_flat), 34962)
        position_accessor = self.add_accessor(
            position_view, 5126, len(transformed), "VEC3", minimum, maximum
        )

        normal_view = self.add_binary(struct.pack("<" + "f" * len(normal_flat), *normal_flat), 34962)
        normal_accessor = self.add_accessor(normal_view, 5126, len(transformed), "VEC3")

        uv_view = self.add_binary(struct.pack("<" + "f" * len(uv_flat), *uv_flat), 34962)
        uv_accessor = self.add_accessor(uv_view, 5126, len(transformed), "VEC2")

        tangent_view = self.add_binary(
            struct.pack("<" + "f" * len(tangent_flat), *tangent_flat), 34962
        )
        tangent_accessor = self.add_accessor(tangent_view, 5126, len(transformed), "VEC4")

        indices = source.indices
        index_view = self.add_binary(
            struct.pack("<" + "I" * len(indices), *indices), 34963
        )
        index_accessor = self.add_accessor(
            index_view, 5125, len(indices), "SCALAR", [min(indices)], [max(indices)]
        )

        self.meshes.append(
            {
                "name": name,
                "primitives": [
                    {
                        "attributes": {
                            "POSITION": position_accessor,
                            "NORMAL": normal_accessor,
                            "TEXCOORD_0": uv_accessor,
                            "TANGENT": tangent_accessor,
                        },
                        "indices": index_accessor,
                        "material": self.material_index,
                        "mode": 4,
                    }
                ],
            }
        )
        self.expected_mesh_stats[name] = (len(transformed), source.triangle_count)
        return len(self.meshes) - 1

    def add_samples(self, parent: int, prefix: str, min_position, min_rotation, max_position, max_rotation):
        min_name = prefix + "Min"
        max_name = prefix + "Max"
        self.add_node(min_name, parent, min_position, min_rotation)
        self.add_node(max_name, parent, max_position, max_rotation)
        self.response_samples.update((min_name, max_name))
        return min_name, max_name

    def response(self, component_id: str, property_name: str, target: str, samples):
        min_name, max_name = samples
        key = {
            "button": "pressed",
            "xAxis": "x_axis",
            "yAxis": "y_axis",
        }[property_name]
        return {
            key: {
                "componentProperty": property_name,
                "states": ["default", "touched", "pressed"],
                "valueNodeProperty": "transform",
                "valueNodeName": target,
                "minNodeName": min_name,
                "maxNodeName": max_name,
            }
        }

    def add_animated(self, component_name: str, source: ObjMesh, pivot_model, component_id, property_name, target_name, min_pos, min_rot, max_pos, max_rot, parent=None):
        pivot_grip = self.point_to_grip(pivot_model)
        if parent is None:
            parent = self.add_node(component_name + "Pivot", self.root, pivot_grip)
        target_index = self.add_node(target_name, parent, min_pos, min_rot)
        samples = self.add_samples(parent, target_name, min_pos, min_rot, max_pos, max_rot)
        mesh_index = self.add_mesh(component_name, source, pivot_model)
        self.add_node(component_name + "Mesh", target_index, mesh=mesh_index)
        if target_name in self.response_nodes:
            fail(f"target node {target_name} is reused")
        self.response_nodes.add(target_name)
        self.component_nodes[component_id] = self.nodes[parent]["name"]
        return self.response(component_id, property_name, target_name, samples), target_index

    def add_component_mesh(self, component_name: str, source: ObjMesh, parent: int):
        mesh_index = self.add_mesh(component_name, source)
        self.add_node(component_name + "Mesh", parent, mesh=mesh_index)

    def write(self, path: Path, images: list[bytes], image_names: list[str], kd_factor):
        if len(images) not in (2, 3):
            fail("expected base-color and packed material maps")
        image_defs = [self.add_image(name, data) for name, data in zip(image_names, images)]
        sampler_index = 0
        texture_defs = [{"sampler": sampler_index, "source": i} for i in range(len(images))]
        metallic_roughness_index = 1
        occlusion_index = 1 if len(images) == 2 else 2
        material = {
            "name": "Steam Frame Controller",
            "pbrMetallicRoughness": {
                "baseColorFactor": [float(kd_factor)] * 3 + [1.0],
                "baseColorTexture": {"index": 0},
                "metallicFactor": 1.0,
                "roughnessFactor": 1.0,
                "metallicRoughnessTexture": {"index": metallic_roughness_index},
            },
            "occlusionTexture": {"index": occlusion_index, "strength": 1.0},
            "doubleSided": False,
        }
        document = {
            "asset": {"version": "2.0", "generator": "steam-frame-wivrn-converter"},
            "scene": 0,
            "scenes": [{"name": "Steam Frame Controller", "nodes": self.roots}],
            "nodes": self.nodes,
            "meshes": self.meshes,
            "materials": [material],
            "textures": texture_defs,
            "images": image_defs,
            "samplers": [
                {
                    "magFilter": 9729,
                    "minFilter": 9987,
                    "wrapS": 10497,
                    "wrapT": 10497,
                }
            ],
            "bufferViews": self.buffer_views,
            "accessors": self.accessors,
            "buffers": [{"byteLength": len(self.binary)}],
            "extras": {
                "source": "Steam Frame controller render model",
                "hand": self.hand,
                "gripTransformBaked": True,
                "meshStats": self.expected_mesh_stats,
            },
        }
        json_bytes = json.dumps(document, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        while len(json_bytes) & 3:
            json_bytes += b" "
        binary = bytes(self.binary)
        binary_chunk = binary + b"\x00" * ((-len(binary)) & 3)
        total_length = 12 + 8 + len(json_bytes) + 8 + len(binary_chunk)
        glb = (
            struct.pack("<4sII", b"glTF", 2, total_length)
            + struct.pack("<I4s", len(json_bytes), b"JSON")
            + json_bytes
            + struct.pack("<I4s", len(binary_chunk), b"BIN\x00")
            + binary_chunk
        )
        path.write_bytes(glb)
        return document, len(glb)


def read_material_factor(hand_dir: Path, hand: str):
    material_path = hand_dir / f"frame_controller_{hand}.mtl"
    factor = 0.48
    if material_path.exists():
        for line in material_path.read_text(encoding="utf-8", errors="replace").splitlines():
            fields = line.split()
            if len(fields) == 4 and fields[0] == "Kd":
                factor = float(fields[1])
                if not all(math.isfinite(float(x)) for x in fields[1:4]):
                    fail(f"{material_path}: non-finite Kd")
                if max(abs(float(x) - factor) for x in fields[1:4]) > 1e-6:
                    fail(f"{material_path}: expected grayscale Kd factor")
                break
    return factor


def load_images(hand_dir: Path, hand: str):
    prefix = f"frame_controller_{hand}_"
    base = hand_dir / (prefix + "color.png")
    rough = hand_dir / (prefix + "rough.png")
    metal = hand_dir / (prefix + "metal.png")
    ao = hand_dir / (prefix + "ao.png")
    for path in (base, rough, metal, ao):
        if not path.is_file():
            fail(f"required texture is missing: {path}")
    if pack_orm is not None:
        packed = pack_orm(ao, rough, metal)
        return [base.read_bytes(), packed], [
            "base_color",
            "occlusion_roughness_metallic",
        ]
    packed = pack_metallic_roughness(rough, metal)
    return [base.read_bytes(), packed, ao.read_bytes()], [
        "base_color",
        "metallic_roughness",
        "occlusion",
    ]


def create_response(component_id, property_name, target, min_name, max_name):
    key = {"button": "pressed", "xAxis": "x_axis", "yAxis": "y_axis"}[property_name]
    return {
        key: {
            "componentProperty": property_name,
            "states": ["default", "touched", "pressed"],
            "valueNodeProperty": "transform",
            "valueNodeName": target,
            "minNodeName": min_name,
            "maxNodeName": max_name,
        }
    }


def make_component(component_type: str, root_name: str, responses: dict[str, Any], gamepad_indices=None):
    component: dict[str, Any] = {
        "type": component_type,
        "rootNodeName": root_name,
        "visualResponses": responses,
    }
    if gamepad_indices is not None:
        component["gamepadIndices"] = gamepad_indices
    return component


def build_hand(source_root: Path, output: Path, hand: str):
    hand_dir = source_root / f"frame_controller_{hand}"
    metadata_path = hand_dir / f"frame_controller_{hand}.json"
    if not metadata_path.is_file():
        fail(f"source metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    components = metadata.get("components")
    if not isinstance(components, dict):
        fail(f"{metadata_path}: missing components object")

    grip = components.get("openxr_grip", {}).get("component_local")
    if not isinstance(grip, dict):
        fail(f"{metadata_path}: missing openxr_grip component_local")
    grip_origin = vec3(grip["origin"])
    grip_rotation = qeuler_xyz(grip["rotate_xyz"])

    model_dir = hand_dir
    builder = GlbBuilder(hand, grip_origin, grip_rotation, {})
    profile_components: dict[str, Any] = {}
    parsed_meshes: dict[str, ObjMesh] = {}
    included_meshes: list[ObjMesh] = []

    for name, entry in components.items():
        filename = entry.get("filename")
        if not filename:
            continue
        visibility = entry.get("visibility", {})
        if visibility.get("default") is False:
            continue
        path = model_dir / filename
        if not path.is_file():
            fail(f"component OBJ is missing: {path}")
        mesh = parse_obj(path)
        parsed_meshes[name] = mesh
        included_meshes.append(mesh)

    builder.validate_grip_pose_roundtrip(
        position for mesh in included_meshes for position in mesh.positions
    )

    master_path = hand_dir / f"frame_controller_{hand}.obj"
    if master_path.is_file():
        master = parse_obj(master_path)
        source_vertices = sum(mesh.raw_vertex_count for mesh in included_meshes)
        source_triangles = sum(mesh.triangle_count for mesh in included_meshes)
        if source_vertices != master.raw_vertex_count or source_triangles != master.triangle_count:
            fail(
                f"{hand}: component OBJ set is not a complete non-duplicated assembly "
                f"(components: {source_vertices} vertices/{source_triangles} triangles; "
                f"master: {master.raw_vertex_count} vertices/{master.triangle_count} triangles)"
            )

    # Body is the only non-animated static component. Every moving mesh is parented
    # below its own pivot so its transform changes around the declared motion center.
    body = parsed_meshes.get("body")
    if body is None:
        fail(f"{hand}: body OBJ is missing")
    body_index = builder.add_mesh("body", body)
    body_node = builder.add_node("Body", builder.root, mesh=body_index)
    builder.component_nodes["body"] = "Body"

    for name, entry in components.items():
        if name == "body" or name not in parsed_meshes:
            continue
        motion = entry.get("motion")
        if not isinstance(motion, dict):
            # Status LEDs are absent by default; unknown static parts remain visible.
            node_name = name.title().replace("_", "")
            mesh_index = builder.add_mesh(name, parsed_meshes[name])
            builder.add_node(node_name, builder.root, mesh=mesh_index)
            continue

        mesh = parsed_meshes[name]
        if name == "thumbstick":
            motion_rotation = qeuler_xyz(motion.get("rotate_xyz", [0.0, 0.0, 0.0]))
            pivot_model = vec3(motion["center"])
            pivot_grip = builder.point_to_grip(pivot_model)
            pivot_node = builder.add_node("ThumbstickPivot", builder.root, pivot_grip)
            responses: dict[str, Any] = {}

            x_values = motion.get("joystick_rotation_x", [0.0, 0.0])
            y_values = motion.get("joystick_rotation_y", [0.0, 0.0])
            if len(x_values) != 2 or len(y_values) != 2:
                fail(f"{hand}: thumbstick rotation must have min/max values")
            basis_grip = builder.orientation_to_grip(motion_rotation)

            # The runtime probe shows xAxis tilts around plane-local +Z.
            x_min_q = qnormalize(qmul(qmul(basis_grip, qaxis((0, 0, 1), float(x_values[0]))), qconjugate(basis_grip)))
            x_max_q = qnormalize(qmul(qmul(basis_grip, qaxis((0, 0, 1), float(x_values[1]))), qconjugate(basis_grip)))
            x_target = builder.add_node("ThumbstickXValue", pivot_node)
            x_samples = builder.add_samples(pivot_node, "ThumbstickX", (0, 0, 0), x_min_q, (0, 0, 0), x_max_q)
            x_response = create_response("xr-standard-thumbstick", "xAxis", "ThumbstickXValue", *x_samples)
            responses.update(x_response)

            y_target = builder.add_node("ThumbstickYValue", x_target)
            # WiVRn reverses Y before interpolating: +Y selects min. The runtime
            # min pose is +20 degrees about -plane-X, so use -X with [20,-20].
            y_min_q = qnormalize(qmul(qmul(basis_grip, qaxis((-1, 0, 0), float(y_values[0]))), qconjugate(basis_grip)))
            y_max_q = qnormalize(qmul(qmul(basis_grip, qaxis((-1, 0, 0), float(y_values[1]))), qconjugate(basis_grip)))
            y_samples = builder.add_samples(x_target, "ThumbstickY", (0, 0, 0), y_min_q, (0, 0, 0), y_max_q)
            responses.update(create_response("xr-standard-thumbstick", "yAxis", "ThumbstickYValue", *y_samples))

            press = builder.direction_to_grip(motion.get("press_translate", [0.0, 0.0, 0.0]))
            press_value = float(vnorm(motion.get("press_translate", [0.0, 0.0, 0.0])))
            if press_value > 0:
                press = vscale(press, press_value)
            click_target = builder.add_node("ThumbstickClickValue", y_target)
            click_samples = builder.add_samples(
                y_target, "ThumbstickClick", (0, 0, 0), IDENTITY,
                press, IDENTITY
            )
            responses.update(create_response("xr-standard-thumbstick", "button", "ThumbstickClickValue", *click_samples))
            mesh_index = builder.add_mesh("thumbstick", mesh, pivot_model)
            builder.add_node("ThumbstickMesh", click_target, mesh=mesh_index)
            builder.component_nodes["xr-standard-thumbstick"] = "ThumbstickPivot"
            profile_components["xr-standard-thumbstick"] = make_component(
                "thumbstick",
                "ThumbstickPivot",
                responses,
                {"button": 3, "xAxis": 2, "yAxis": 3},
            )
            builder.response_nodes.update(("ThumbstickXValue", "ThumbstickYValue", "ThumbstickClickValue"))
            continue

        if name == "dpad":
            pivot_model = vec3(motion["center"])
            pivot_node = builder.add_node("DpadPivot", builder.root, builder.point_to_grip(pivot_model))
            rotation = qeuler_xyz(motion.get("rotate_xyz", [0.0, 0.0, 0.0]))
            basis_grip = builder.orientation_to_grip(rotation)
            x_values = motion.get("joystick_rotation_x", [0.0, 0.0])
            y_values = motion.get("joystick_rotation_y", [0.0, 0.0])
            push = motion.get("press_translate", [0.0, 0.0, 0.0])
            push_grip = builder.vector_to_grip(push)
            parent = pivot_node
            direction_values = {
                "up": (qaxis((-1, 0, 0), float(y_values[0])), push_grip),
                "down": (qaxis((-1, 0, 0), float(y_values[1])), push_grip),
                "left": (qaxis((0, 0, 1), float(x_values[0])), push_grip),
                "right": (qaxis((0, 0, 1), float(x_values[1])), push_grip),
            }
            responses_by_id: dict[str, Any] = {}
            target_names: dict[str, str] = {}
            for direction in ("up", "down", "left", "right"):
                component_id = DPAD_IDS[direction]
                target_name = "Dpad" + direction.title() + "Value"
                local_q, max_translation = direction_values[direction]
                rotation_grip = qnormalize(
                    qmul(qmul(basis_grip, local_q), qconjugate(basis_grip))
                )
                target = builder.add_node(target_name, parent)
                samples = builder.add_samples(
                    parent, "Dpad" + direction.title(), (0, 0, 0), IDENTITY,
                    max_translation, rotation_grip
                )
                responses_by_id[component_id] = create_response(
                    component_id, "button", target_name, *samples
                )
                target_names[direction] = target_name
                builder.response_nodes.add(target_name)
                builder.component_nodes[component_id] = "DpadPivot"
                parent = target
            mesh_index = builder.add_mesh("dpad", mesh, pivot_model)
            builder.add_node("DpadMesh", parent, mesh=mesh_index)
            for direction, component_id in DPAD_IDS.items():
                profile_components[component_id] = make_component(
                    "button",
                    "DpadPivot",
                    responses_by_id[component_id],
                )
            continue

        component_id = COMPONENT_IDS.get(name)
        if not component_id:
            fail(f"{hand}: no profile component mapping for moving OBJ {name}")
        kind = "trigger" if name == "trigger" else (
            "squeeze" if name == "button_grip" else "button"
        )
        if name in ("trigger", "button_grip", "bumper"):
            pivot_model = vec3(motion["pivot"])
            axis = vec3(motion["axis"])
            angle_values = motion.get("value_mapping", [0.0, 0.0])
            if len(angle_values) != 2:
                fail(f"{hand}: {name} value_mapping must have min/max values")
            min_q = qaxis(builder.direction_to_grip(axis), float(angle_values[0]))
            max_q = qaxis(builder.direction_to_grip(axis), float(angle_values[1]))
            min_p = (0.0, 0.0, 0.0)
            max_p = (0.0, 0.0, 0.0)
        else:
            local = entry.get("component_local", {})
            pivot_model = vec3(local.get("origin", [0.0, 0.0, 0.0]))
            axis = builder.direction_to_grip(motion.get("axis", [0.0, 1.0, 0.0]))
            values = motion.get("value_mapping", [0.0, 0.0])
            if len(values) != 2:
                fail(f"{hand}: {name} value_mapping must have min/max values")
            min_q = IDENTITY
            max_q = IDENTITY
            min_p = (0.0, 0.0, 0.0)
            max_p = vscale(axis, float(values[1]))

        target_name = name.title().replace("_", "") + "Value"
        response, _ = builder.add_animated(
            name,
            mesh,
            pivot_model,
            component_id,
            "button",
            target_name,
            min_p,
            min_q,
            max_p,
            max_q,
        )
        root_name = builder.component_nodes[component_id]
        profile_components[component_id] = make_component(
            kind,
            root_name,
            response,
            {"button": {"xr-standard-trigger": 0, "xr-standard-squeeze": 1}.get(component_id, 0)}
            if component_id in ("xr-standard-trigger", "xr-standard-squeeze")
            else None,
        )

    images, image_names = load_images(hand_dir, hand)
    output.mkdir(parents=True, exist_ok=True)
    glb_path = output / f"frame_controller_{hand}.glb"
    document, size = builder.write(
        glb_path, images, image_names, read_material_factor(hand_dir, hand)
    )

    profile_layout = {
        "selectComponentId": "xr-standard-trigger",
        "components": profile_components,
        "gamepadMapping": "xr-standard",
        "rootNodeName": "FrameController" + hand.title(),
        "assetPath": glb_path.name,
    }
    return profile_layout, document, glb_path, size, builder, included_meshes


def make_profile(layouts):
    return {
        "profileId": PROFILE_ID,
        "fallbackProfileIds": ["generic-trigger-squeeze-thumbstick"],
        "layouts": layouts,
    }


def validate_generated(output: Path, profile: dict[str, Any], builds):
    # Check profile links and ensure every response resolves to a unique GLB target.
    for hand in ("left", "right"):
        layout = profile["layouts"][hand]
        glb_path = output / layout["assetPath"]
        data = glb_path.read_bytes()
        if data[:4] != b"glTF":
            fail(f"{glb_path}: not a GLB file")
        magic, version, total = struct.unpack_from("<4sII", data, 0)
        if magic != b"glTF" or version != 2 or total != len(data):
            fail(f"{glb_path}: invalid GLB header")
        json_length, chunk_type = struct.unpack_from("<I4s", data, 12)
        if chunk_type != b"JSON":
            fail(f"{glb_path}: first GLB chunk is not JSON")
        document = json.loads(data[20 : 20 + json_length])
        names = [node.get("name") for node in document.get("nodes", [])]
        if len(names) != len(set(names)):
            fail(f"{glb_path}: node names are not unique")
        accessors = document["accessors"]
        for mesh in document["meshes"]:
            for primitive in mesh["primitives"]:
                attrs = primitive["attributes"]
                pos = accessors[attrs["POSITION"]]
                normal = accessors[attrs["NORMAL"]]
                uv = accessors[attrs["TEXCOORD_0"]]
                tangent = accessors[attrs["TANGENT"]]
                if not (pos["count"] == normal["count"] == uv["count"] == tangent["count"]):
                    fail(f"{glb_path}: vertex accessor counts do not agree")
                if pos.get("min") is None or pos.get("max") is None:
                    fail(f"{glb_path}: POSITION accessor is missing bounds")
                index_accessor = accessors[primitive["indices"]]
                if index_accessor["count"] % 3:
                    fail(f"{glb_path}: index count is not a triangle list")
        for component_id, component in layout["components"].items():
            for response in component["visualResponses"].values():
                target = response["valueNodeName"]
                minimum = response["minNodeName"]
                maximum = response["maxNodeName"]
                if target not in names or minimum not in names or maximum not in names:
                    fail(f"{glb_path}: response node reference does not exist: {target}")
        built = builds[hand]
        expected_vertices = sum(len(mesh.positions) for mesh in built["meshes"])
        expected_triangles = sum(mesh.triangle_count for mesh in built["meshes"])
        actual_vertices = sum(mesh_stat[0] for mesh_stat in document["extras"]["meshStats"].values())
        actual_triangles = sum(mesh_stat[1] for mesh_stat in document["extras"]["meshStats"].values())
        if actual_vertices != expected_vertices or actual_triangles != expected_triangles:
            fail(f"{glb_path}: mesh counts do not match source component OBJs")
        if len(document["images"]) not in (2, 3):
            fail(f"{glb_path}: expected embedded base-color and packed material images")
        if "normalTexture" in document["materials"][0]:
            fail(f"{glb_path}: speculative normalTexture must not be enabled")
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        "--input-dir",
        dest="source_root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help=f"SteamVR render-model directory (default: {DEFAULT_SOURCE_ROOT})",
    )
    parser.add_argument(
        "--output",
        "--output-dir",
        dest="output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"directory for generated GLBs and profile.json (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args(argv)
    source_root = args.source_root.expanduser()
    output = args.output.expanduser()

    layouts = {}
    builds = {}
    for hand in ("left", "right"):
        layout, document, path, size, builder, source_meshes = build_hand(
            source_root, output, hand
        )
        layouts[hand] = layout
        builds[hand] = {"meshes": source_meshes, "builder": builder}
        print(
            f"{hand}: wrote {path} ({size:,} bytes; "
            f"{sum(len(mesh.positions) for mesh in source_meshes):,} vertices, "
            f"{sum(mesh.triangle_count for mesh in source_meshes):,} triangles)"
        )

    profile = make_profile(layouts)
    profile_path = output / "profile.json"
    profile_path.write_text(
        json.dumps(profile, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    validate_generated(output, profile, builds)
    print(f"profile: wrote {profile_path}")
    print("validation: GLB headers, profile node references, and source mesh counts passed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"convert-frame-controller: {error}", file=sys.stderr)
        raise SystemExit(2) from None
