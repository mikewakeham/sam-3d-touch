"""CPU coordinate contract audit and exact signed-axis occupancy rotations."""
import ast
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def rotations():
    return {
        "identity": np.eye(3, dtype=int),
        "z90": np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
        "z180": np.diag([-1, -1, 1]),
        "z270": np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]),
        "x90_axis_control": np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]),
        "x270_axis_control": np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]),
    }


def rotate_grid(grid, rotation):
    """Rotate xyz grid about its center: column-vector coordinates p' = R p.

    Reindex physical occupancy, never neural latent channels. Signed axis
    permutations map voxel centers exactly, with no interpolation or cropping.
    """
    grid = np.asarray(grid)
    r = np.asarray(rotation)
    assert grid.ndim == 3 and len(set(grid.shape)) == 1
    assert np.isin(r, [-1, 0, 1]).all()
    assert np.array_equal(r @ r.T, np.eye(3)) and np.isclose(np.linalg.det(r), 1)
    axes = np.argmax(np.abs(r), axis=1)
    result = grid.transpose(tuple(axes))
    for i, axis in enumerate(axes):
        if r[i, axis] < 0:
            result = np.flip(result, i)
    return np.ascontiguousarray(result)


def audit():
    source = REPO / "sam3d_objects/pipeline/inference_utils.py"
    tree = ast.parse(source.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "voxelize_mesh")
    expression = next(n.value for n in function.body if isinstance(n, ast.Assign)
                      and isinstance(n.value, ast.BinOp) and isinstance(n.value.op, ast.MatMult))
    assert isinstance(expression.right, ast.Attribute) and expression.right.attr == "T"
    stock = np.array(ast.literal_eval(expression.right.value.args[0]))
    np.testing.assert_array_equal(stock, rotations()["x90_axis_control"])
    post = REPO / "sam3d_objects/model/backbone/tdfy_dit/utils/postprocessing_utils.py"
    matches = [n for n in ast.walk(ast.parse(post.read_text())) if isinstance(n, ast.BinOp)
               and isinstance(n.op, ast.MatMult) and isinstance(n.right, ast.Call)
               and isinstance(n.right.func, ast.Attribute) and n.right.func.attr == "array"]
    assert any(np.array_equal(np.array(ast.literal_eval(n.right.args[0])), stock) for n in matches)
    rows = []
    for path in sorted((REPO / "data_generation/objaverse-dexonomy/generated_data").glob("*/object_transform.npz")):
        with np.load(path, allow_pickle=False) as data:
            transform = data["T_normalized_from_source"].astype(float)
        linear = transform[:3, :3]
        scale = np.linalg.norm(linear[:, 0])
        error = float(np.max(np.abs(linear / scale - stock)))
        assert error < 1e-6
        rows.append({"object_id": path.parent.name, "scale": scale,
                     "rotation_max_abs_error": error,
                     "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    assert rows
    # Verify indexing against physical voxel centers, including signs, on every
    # proper axis permutation, not just the six proposed probe candidates.
    grid = np.arange(7 ** 3).reshape(7, 7, 7)
    coords = np.indices(grid.shape).reshape(3, -1).T
    tested = 0
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([-1, 1], repeat=3):
            r = np.eye(3, dtype=int)[list(perm)] * np.array(signs)[:, None]
            if round(np.linalg.det(r)) != 1:
                continue
            rotated = rotate_grid(grid, r)
            destination = ((coords - 3) @ r.T + 3).astype(int)
            np.testing.assert_array_equal(rotated[tuple(destination.T)], grid.flatten())
            np.testing.assert_array_equal(rotate_grid(rotated, r.T), grid)
            tested += 1
    result = {"source_sha256": {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in [source, post, Path(__file__)]},
              "stock_column_rotation": stock.tolist(), "saved_transforms": rows,
              "objects": len(rows), "max_rotation_error": max(r["rotation_max_abs_error"] for r in rows),
              "proper_grid_rotations_verified": tested,
              "conclusion": "Saved source-to-render rotation equals stock voxelizer rotation. Applying stock rotation again to the already transformed mesh would double it.",
              "limits": "Checks linear rotation and export inverse, not source semantic front direction, source mesh reconstruction, or neural compatibility."}
    (HERE / "frame_contract_results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ["objects", "max_rotation_error", "proper_grid_rotations_verified", "conclusion"]}, indent=2))


if __name__ == "__main__":
    audit()
