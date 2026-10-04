# SPDX-License-Identifier: GPL-3.0-or-later
"""Small stdlib-only PNG helpers for Frame controller render models.

The source material maps are single-channel 8-bit PNGs.  ``pack_orm`` combines
them into the channel layout expected by glTF metallic-roughness materials:
occlusion in R, roughness in G, metallic in B, and opaque alpha.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path
from typing import Union


PathLike = Union[str, Path]
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _paeth(left: int, above: int, upper_left: int) -> int:
    estimate = left + above - upper_left
    left_distance = abs(estimate - left)
    above_distance = abs(estimate - above)
    diagonal_distance = abs(estimate - upper_left)
    if left_distance <= above_distance and left_distance <= diagonal_distance:
        return left
    if above_distance <= diagonal_distance:
        return above
    return upper_left


def _read_png_scalar(path: PathLike) -> tuple[int, int, bytes]:
    """Decode one 8-bit, non-interlaced PNG as a scalar channel.

    Grayscale and grayscale-alpha use the grayscale sample. RGB and RGBA use
    their red sample, which is the conventional scalar channel for packed
    material maps. Palette, indexed, 16-bit, and interlaced PNGs are rejected.
    """
    path = Path(path)
    try:
        data = path.read_bytes()
    except OSError as error:
        raise ValueError(f"cannot read PNG {path}: {error}") from error
    if not data.startswith(_PNG_SIGNATURE):
        raise ValueError(f"not a PNG file: {path}")

    width = height = bit_depth = color_type = interlace = None
    idat = bytearray()
    offset = len(_PNG_SIGNATURE)
    saw_iend = False
    saw_idat = False
    idat_ended = False

    while offset < len(data):
        if offset + 12 > len(data):
            raise ValueError(f"truncated PNG chunk in {path}")
        length = struct.unpack_from(">I", data, offset)[0]
        chunk_type = data[offset + 4 : offset + 8]
        chunk_end = offset + 12 + length
        if chunk_end > len(data):
            raise ValueError(f"truncated {chunk_type.decode('ascii', 'replace')} chunk in {path}")
        chunk_data = data[offset + 8 : offset + 8 + length]
        expected_crc = struct.unpack_from(">I", data, offset + 8 + length)[0]
        actual_crc = zlib.crc32(chunk_type)
        actual_crc = zlib.crc32(chunk_data, actual_crc) & 0xFFFFFFFF
        if actual_crc != expected_crc:
            raise ValueError(f"bad CRC in {chunk_type.decode('ascii', 'replace')} chunk of {path}")

        if chunk_type == b"IHDR":
            if width is not None or length != 13:
                raise ValueError(f"invalid PNG IHDR in {path}")
            width, height, bit_depth, color_type, compression, filtering, interlace = struct.unpack(
                ">IIBBBBB", chunk_data
            )
            if compression != 0 or filtering != 0:
                raise ValueError(f"unsupported PNG compression or filter method in {path}")
        elif chunk_type == b"IDAT":
            if idat_ended:
                raise ValueError(f"non-contiguous IDAT chunks in {path}")
            saw_idat = True
            idat.extend(chunk_data)
        elif chunk_type == b"IEND":
            if length != 0:
                raise ValueError(f"invalid PNG IEND in {path}")
            saw_iend = True
            offset = chunk_end
            break
        else:
            if saw_idat:
                idat_ended = True
            # Unknown critical chunks affect image interpretation and cannot
            # safely be ignored. Ancillary chunks do not affect pixel values.
            if chunk_type[0] & 0x20 == 0 and chunk_type not in (b"PLTE",):
                raise ValueError(f"unsupported critical PNG chunk {chunk_type!r} in {path}")
        if chunk_type != b"IDAT" and saw_idat:
            idat_ended = True
        offset = chunk_end

    if not saw_iend or width is None or height is None or not saw_idat:
        raise ValueError(f"incomplete PNG file: {path}")
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid PNG dimensions {width}x{height} in {path}")
    if bit_depth != 8 or color_type not in (0, 2, 4, 6) or interlace != 0:
        raise ValueError(
            f"unsupported PNG format in {path}: bit depth {bit_depth}, color type {color_type}, "
            f"interlace {interlace}; expected 8-bit grayscale/RGB, non-interlaced"
        )

    channels = {0: 1, 2: 3, 4: 2, 6: 4}[color_type]
    bytes_per_pixel = channels
    row_bytes = width * channels
    try:
        decoded = zlib.decompress(idat)
    except zlib.error as error:
        raise ValueError(f"invalid PNG image data in {path}: {error}") from error
    expected_size = height * (row_bytes + 1)
    if len(decoded) != expected_size:
        raise ValueError(
            f"unexpected PNG data size in {path}: got {len(decoded)}, expected {expected_size}"
        )

    scalar = bytearray(width * height)
    previous = bytearray(row_bytes)
    source_offset = 0
    target_offset = 0
    for _ in range(height):
        filter_type = decoded[source_offset]
        source_offset += 1
        filtered = decoded[source_offset : source_offset + row_bytes]
        source_offset += row_bytes
        row = bytearray(row_bytes)
        if filter_type == 0:
            row[:] = filtered
        elif filter_type in (1, 2, 3, 4):
            for index, value in enumerate(filtered):
                left = row[index - bytes_per_pixel] if index >= bytes_per_pixel else 0
                above = previous[index]
                upper_left = previous[index - bytes_per_pixel] if index >= bytes_per_pixel else 0
                if filter_type == 1:
                    predictor = left
                elif filter_type == 2:
                    predictor = above
                elif filter_type == 3:
                    predictor = (left + above) // 2
                else:
                    predictor = _paeth(left, above, upper_left)
                row[index] = (value + predictor) & 0xFF
        else:
            raise ValueError(f"unsupported PNG filter {filter_type} in {path}")

        if channels == 1:
            scalar[target_offset : target_offset + width] = row
        elif channels == 2:
            scalar[target_offset : target_offset + width] = row[0::2]
        else:
            scalar[target_offset : target_offset + width] = row[0::channels]
        target_offset += width
        previous = row

    return width, height, bytes(scalar)


def _png_chunk(chunk_type: bytes, chunk_data: bytes) -> bytes:
    crc = zlib.crc32(chunk_type)
    crc = zlib.crc32(chunk_data, crc) & 0xFFFFFFFF
    return struct.pack(">I", len(chunk_data)) + chunk_type + chunk_data + struct.pack(">I", crc)


def pack_orm(ao_path: PathLike, rough_path: PathLike, metal_path: PathLike) -> bytes:
    """Return an RGBA PNG with AO in R, roughness in G, metalness in B.

    The three inputs must be 8-bit, non-interlaced PNGs of identical size.
    Their scalar values are preserved byte-for-byte; alpha is set to 255.
    """
    width, height, ao = _read_png_scalar(ao_path)
    rough_width, rough_height, roughness = _read_png_scalar(rough_path)
    metal_width, metal_height, metalness = _read_png_scalar(metal_path)
    if (rough_width, rough_height) != (width, height):
        raise ValueError(
            f"roughness dimensions {rough_width}x{rough_height} do not match AO {width}x{height}"
        )
    if (metal_width, metal_height) != (width, height):
        raise ValueError(
            f"metalness dimensions {metal_width}x{metal_height} do not match AO {width}x{height}"
        )

    row_bytes = width * 4
    raw = bytearray(height * (row_bytes + 1))
    for y in range(height):
        raw_row_start = y * (row_bytes + 1)
        raw[raw_row_start] = 0  # PNG filter None
        out = raw_row_start + 1
        src_start = y * width
        for x in range(width):
            src = src_start + x
            raw[out] = ao[src]
            raw[out + 1] = roughness[src]
            raw[out + 2] = metalness[src]
            raw[out + 3] = 255
            out += 4

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    return (
        _PNG_SIGNATURE
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", zlib.compress(raw, 6))
        + _png_chunk(b"IEND", b"")
    )
