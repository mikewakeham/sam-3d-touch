# Data pipeline and touch format

This document describes the current executable data path, not the older format-v2
README. Paths are relative to
`sam-3d-touch-data/objaverse-dexonomy/` unless stated otherwise.

## End-to-end object flow

For one object, the intended current pipeline is:

```text
objects/<object_id>/
  model.obj + material.mtl + material_0.png
          |
          v
render_blender.py
  import/axis conversion -> normalize object -> 16 camera views
          |
          +--> object_transform.npz
          +--> views/<view>/image.png
          +--> views/<view>/depth.npy
          +--> views/<view>/camera.npz
          |
          v
make_data.py::depth_to_pointmap
          +--> views/<view>/pointmap.npy
          |
          v
sample_touch.py
  one deterministic dense surface pool per object
  + per-view visibility/contact selection
          +--> views/<view>/touches.npz
          |
          v
make_data.py manifest append
          +--> generated_data/samples.jsonl
```

The remaining object-level Stage 1 branch is:

```text
objects/<object_id>/model.obj
  + generated_data/<object_id>/object_transform.npz
          |
          v
apply T_normalized_from_source exactly once
          |
          v
fixed-bound 64^3 surface occupancy in normalized object coordinates
          |
          v
frozen SS encoder, posterior sampling disabled
          |
          +--> generated_data/<object_id>/target_latent.npz
```

`make_data.py::make_stage1_target` currently returns `None`. Its docstring sketches
the correct object-level placement, but no target file is created. Target generation
should be a separate offline pass so the VAE encoder is loaded once and never run in
each training iteration or rendering worker.

## Object discovery and splits

### Required source files

`make_data.py::get_objects` accepts an object only when all three are present:

- `model.obj`
- `material.mtl`
- `material_0.png`

The source OBJ directory is not included in the local snapshot.

### Deterministic object-level split

Defaults in `make_data.py` include:

- seed: `29`
- train fraction: `0.8`
- validation fraction: `0.1`
- remainder: test
- views per object: `16`
- resolution: `768`
- camera radius: `2.0`
- field of view: `40` degrees

Object IDs are shuffled with the configured seed and assigned to splits, then IDs
within each split are sorted. The inspected `splits.json` has 1024 objects:

| Split | Objects |
|---|---:|
| train | 819 |
| validation | 102 |
| test | 103 |

There is no split overlap. Because the split is at object level, views of one object
do not intentionally cross splits.

### Per-object and per-view seeds

- The renderer's object seed is `seed XOR int(object_id[:8], 16)`.
- The dense touch surface pool uses seed parts
  `[seed, int(object_id[:8], 16)]`.
- Per-view contact selection uses
  `[seed, int(object_id[:8], 16), int(view_id)]`.

This makes the master pool repeat across views and standalone invocations when the
mesh, code, software versions, density, and seed remain unchanged. Contact centers
still vary by view because visibility and the per-view RNG differ.

## Rendering and object normalization

### Import and normalization

`render_blender.py` uses Blender's OBJ import and explicitly composes the OBJ axis
conversion returned by:

```python
axis_conversion(from_forward="-Z", from_up="Y")
```

It repairs/recalculates normals, finds the imported scene's axis-aligned bounding
box, scales the largest extent to exactly 1, and centers the bounding box at the
origin. It saves the complete source-to-normalized transform, including import-axis
conversion, as:

```text
object_transform.npz["T_normalized_from_source"]  # [4,4]
```

This transform is essential. Touch sampling loads the original OBJ and applies it;
using an older transform that omits the import rotation would misalign touch geometry
with rendered images/cameras.

The transform's axis matrix is:

```text
[[1, 0,  0],
 [0, 0, -1],
 [0, 1,  0]]
```

It maps source `(x,y,z)` to normalized-object `(x,-z,y)` before uniform scale and
translation. Some released comments describe this rotation in the opposite direction;
the executable matrix and the saved renderer transform are the authority.

For the local sample object, the saved transform is approximately:

```text
[[ 0.697942,  0,         0,         0],
 [ 0,         0,        -0.697942,  0],
 [ 0,         0.697942,  0,         0],
 [ 0,         0,         0,         1]]
```

The zero translation means that object's source bounding box was already centered
under the composed import transform. This is an example, not a global guarantee.

### Camera and outputs

Rendering uses Cycles with transparent RGBA output and 32-bit OpenEXR depth internally.
The code configures CUDA, 128 samples, near/far clips `0.1`/`100`, and a camera looking
at the object origin. Camera positions are deterministically distributed over views
using a low-discrepancy-like azimuth/elevation sequence plus a seeded offset.

For square resolution `W = H = resolution`, the intrinsics are:

```text
f = 0.5 * resolution / tan(FOV / 2)
cx = cy = (resolution - 1) / 2
K = [[f, 0, cx], [0, f, cy], [0, 0, 1]]
```

The local `768 x 768`, 40-degree sample has:

```text
fx = fy = 1055.0314
cx = cy = 383.5
```

Each view contains:

- `image.png`: uint8 RGBA;
- `depth.npy`: float32 camera-axis Z, with invalid pixels represented as `NaN`;
- `camera.npz["K"]`: `[3,3]` intrinsics;
- `camera.npz["T_camera_from_object"]`: `[4,4]` normalized-object to OpenCV-camera
  transform.

The saved camera frame at this point is OpenCV-like:

- +X: image right
- +Y: image down
- +Z: forward

The renderer obtains it by applying `diag(1, -1, -1)` to Blender camera coordinates.

### Alpha is not exactly a depth-validity mask

RGBA alpha has antialiased edge values. In the real view-000 sample:

- pixels with alpha greater than zero: 28,418;
- pixels with finite depth: 27,622.

Therefore alpha-positive and finite-depth masks should not be assumed identical.
The base image pipeline binarizes alpha with `alpha > 0`; the pointmap pipeline's
current preprocessing path can preserve fractional alpha values. Training must match
the specific released pipeline/config rather than mixing these behaviors accidentally.

## Pointmap generation and coordinate frames

`make_data.py::depth_to_pointmap` first unprojects depth in OpenCV axes:

```text
x_cv = (u - cx) * z / fx
y_cv = (v - cy) * z / fy
z_cv = depth
```

It then flips X and Y:

```text
pointmap = (-x_cv, -y_cv, z_cv)
```

Thus `pointmap.npy` uses the SAM/PyTorch3D camera convention:

- +X: left
- +Y: up
- +Z: forward

Invalid depth pixels become `(NaN, NaN, NaN)`.

Define:

```text
S = diag(-1, -1, 1, 1)
T_sam_from_object = S @ T_camera_from_object
```

The same `S` is used by touch packing. This is the key cross-modal frame invariant:
raw touch camera coordinates and `pointmap.npy` coordinates agree before checkpoint
pointmap normalization.

In real view 000:

- depth and pointmap Z are exactly equal, including their finite mask;
- finite depth spans about `1.510` to `2.205`;
- pointmap X spans about `-0.060` to `0.069`;
- pointmap Y spans about `-0.482` to `0.494`.

The asymmetric apparent ranges are view/object specific.

## Stage 1 shape-target contract

### One object target, not one view target

The Stage 1 shape target is independent of camera view, RGB, pointmap, and touch
variant. Every accepted view of one object references the same file:

```text
generated_data/<object_id>/target_latent.npz
```

The offline generator should iterate unique object IDs from
`generated_data/samples.jsonl`, not every directory under `objects/`. This limits the
target cache to objects that actually have accepted training samples and naturally
skips failed/excluded objects.

For each object it requires only:

- `objects/<object_id>/model.obj`;
- `generated_data/<object_id>/object_transform.npz`;
- the official `ss_encoder.yaml`/`ss_encoder.ckpt` contract.

Existing RGB, depth, pointmap, camera, and touch arrays do not need to be regenerated
or modified. A camera and pointmap are needed only for the one-time alignment check.

### Canonical target coordinates

Use these frame names:

```text
S     original OBJ coordinates
O     renderer-normalized object coordinates
C_cv  OpenCV camera coordinates
C_sam SAM/PyTorch3D pointmap coordinates
```

The matrix chain is:

```text
T_O_from_S       = object_transform["T_normalized_from_source"]
T_Ccv_from_O     = camera["T_camera_from_object"]
F                = diag(-1,-1,1,1)
T_Csam_from_O    = F @ T_Ccv_from_O
```

Target occupancy and the encoded shape latent stay in `O`. Load the original OBJ and
apply `T_O_from_S` exactly once. Do not then call the stock
`inference_utils.voxelize_mesh`, because it applies its own axis conversion and may
renormalize, producing a double transform. Do not apply a camera transform or the
`F` camera-axis flip while creating the target.

For validation only, a target voxel center in `O` maps to the pointmap frame by:

```text
p_Csam = T_Csam_from_O @ p_O
```

### Fixed 64-cube occupancy

The selected first-cache policy is fixed-bound Open3D surface voxelization:

```text
resolution = 64
voxel_size = 1/64
min_bound  = (-0.5,-0.5,-0.5)
max_bound  = ( 0.5, 0.5, 0.5)
occupancy shape = [1,64,64,64]
```

Open3D `grid_index` is written directly as occupancy `(x,y,z)`. Voxel index `i` maps
back to its normalized-object center as:

```text
p_O = (i + 0.5) / 64 - 0.5
```

Clip transformed vertices to `[-0.5+1e-6, 0.5-1e-6]` before voxelization and require
all resulting indices to lie in `[0,63]`.

This policy matches SAM's main convenience helper and TRELLIS's released dataset
voxelizer. Meta's private target builder is not released. A later SAM encoder demo
attempts interior filling, but uses Trimesh-local sparse indices as though they were
global 64-cube indices, which can shift an object in the grid. Do not copy that
indexing. Keep the first cache consistently surface-only rather than mixing solid and
surface targets according to mesh watertightness.

### Encoded file

Run the frozen official encoder in eval/inference mode with posterior sampling
disabled:

```text
occupancy [1,1,64,64,64] float32
-> posterior mean [1,8,16,16,16]
-> save mean[0] [8,16,16,16] float32
```

Save `mean`, not sampled `z`, and do not apply Stage 2 mean/std constants or another
latent normalization. The training loader converts one saved mean to generator tokens
with:

```python
shape_target = mean.permute(1, 2, 3, 0).reshape(4096, 8)
```

This flattening is `(x,y,z)` order with `z` varying fastest and exactly inverts the
released inference reshape.

## Touch surface preparation

### Mesh loading and cleanup

`sample_touch.py::load_mesh` loads the source OBJ, applies
`T_normalized_from_source`, and attempts to make geometry suitable for geodesic
distance computation:

- welds coincident geometric vertices created by texture/normal seams;
- discards degenerate faces and unreferenced vertices;
- splits vertex-nonmanifold structures into components;
- rejects excessive fragmentation (more than 32 added components);
- repairs face winding/normals.

This cleanup is not a guarantee of a watertight or semantically correct exterior
surface. Internal shells and disconnected pieces can remain.

`prepare_surface` repeatedly refines the mesh while the current maximum edge exceeds
`0.03` and another fourfold face increase fits under the 250,000-face cap. Because of
the cap, the resulting mesh can still have edges longer than `0.03`.

### Dense master pool

`build_surface` draws:

```text
ceil(mesh.surface_area * density)
```

points by triangle area. Current default density is 200,000 points per squared
normalized-object unit. For each master point, the pool stores:

- point XYZ in normalized object coordinates;
- source face ID;
- barycentric coordinates within that face;
- face normal;
- connected-component ID;
- one fixed uniform `keep_priority` in `[0,1)`.

The pool is area-uniform in expectation, not a Poisson/minimum-spacing set. The
`keep_priority` value belongs to the master point, so the same point keeps the same
priority across overlapping contacts and views.

## Visibility classification

For each view, master points are transformed into the OpenCV camera frame, projected
with `K`, and compared to rendered depth using tolerance `0.005` by default.

Labels are:

| Value | Meaning |
|---:|---|
| `1` | projected point agrees with rendered depth within tolerance; visible |
| `0` | projected point is behind rendered depth by more than tolerance; hidden |
| `-1` | unknown: outside image, behind camera, invalid projection/depth, or in front of the rendered surface |

The final category is important: a point in front of the rendered depth is not called
visible; it is `-1` because it does not agree with the rendered surface.

## Geodesic contacts

Current defaults resolved into the sample files are:

```json
{
  "num_contacts": 32,
  "radius": 0.1,
  "center_method": "geodesic_farthest",
  "tolerance": 0.005,
  "neighborhood_mode": "all"
}
```

### Distances

For each selected center, `geodesic_distances`:

1. runs a `potpourri3d.MeshFastMarchingDistanceSolver` inside the center's connected
   component, using the center as a barycentric source point;
2. interpolates vertex distances back to all master points from barycentric weights;
3. replaces same-face distances with exact Euclidean distances from the center;
4. sets other connected components to infinity.

The current saved `distance_method` is `fast_marching`. The old README's heat-method
and vertex-center description is no longer accurate.

### Center eligibility and selection

Center candidates are always hidden (`visibility == 0`) master-pool points. Centers
are not cleaned-mesh vertices. A selected center therefore has:

- `center_point_ids`: master-pool index;
- `center_face_ids`: containing refined-mesh face;
- `center_barycentric`: its position within that face.

Candidate order is deterministically permuted per view. At each contact selection,
a connected component is sampled with probability proportional to its original
hidden-candidate count among non-exhausted components. Within that chosen component:

- `random` selects the next candidate in seeded order;
- `geodesic_farthest` selects the candidate maximizing distance to the nearest
  already selected center in that component.

This is **not** globally farthest-point sampling across all components because the
component is randomly selected first.

### Neighborhood membership

The geodesic ball includes points with distance at most `radius` within the center's
component. Eligibility then depends on `neighborhood_mode`:

- `"all"`: visible, hidden, and unknown master points may be included;
- `"hidden"`: only hidden master points may be included.

The current generated sample uses `"all"`. It is therefore wrong to describe the
saved neighborhoods as entirely hidden geometry. Only the centers are guaranteed
hidden.

There is no fixed point count, refill, duplication quota, approach-ray constraint,
normal-angle threshold, multiview visibility test, or internal-shell filter. Point
counts vary with surface geometry and radius.

## Contact frames and packing

### Local frame construction

For center face normal `n`, `contact_frame` defines:

- local Z = `n`;
- helper axis = global object-coordinate axis least aligned with `n`;
- local X = normalized `cross(helper, Z)`;
- local Y = `cross(Z, X)`.

The columns of `R_object_from_local` are `[X, Y, Z]`.

This has two consequences:

1. the normal/approach axis is geometrically meaningful;
2. tangent roll is a deterministic synthetic choice tied to normalized object axes,
   not a measured tactile sensor orientation.

The latter can expose canonical object orientation when the full rotation is provided
to a network. It should be an explicit modeling decision, particularly in any
“XYZ-only” ablation.

### Saved transformations

Let:

```text
R_sam_from_object = T_sam_from_object[:3,:3]
R_sam_from_local  = R_sam_from_object @ R_object_from_local
```

The file stores `R_camera_from_local = R_sam_from_local` despite the shorter name
“camera”; it uses SAM camera axes, matching the pointmap.

Neighborhood points and normals are saved in the local frame using row-vector array
convention. Reconstruction is:

```python
points_sam  = points_local @ R_camera_from_local.T + center_camera
normals_sam = normals_local @ R_camera_from_local.T
```

For every audited contact:

- `R.T @ R` is identity to float tolerance;
- determinant is approximately +1;
- the third column of `R_camera_from_local` exactly equals `normals_camera`;
- the center itself occurs once in its neighborhood with local XYZ zero and geodesic
  distance zero.

The rotation and center normal are therefore partially redundant. A model need not
receive both unless the redundancy is intentional.

### What “XYZ-only” can mean

`points_local` alone are not fully normal-independent: the local frame's Z axis came
from the center face normal, and its tangent axes came from the object frame. Possible
ablations should be named precisely:

- local XYZ, no explicit normals/pose;
- local XYZ + center position;
- local XYZ + center position + center normal only;
- local XYZ + full `R_camera_from_local`;
- camera-space XYZ reconstructed from the local arrays;
- local XYZ + per-point normals.

Calling the first option “no normals” is only approximately true because the normal
was used to define the coordinates.

## `touches.npz` format v4

`C` is number of contacts and `P` is the total flattened neighborhood-point count.
Contact `i` occupies slice `offsets[i]:offsets[i+1]`.

| Key | Shape | Dtype | Meaning |
|---|---|---|---|
| `points_local` | `[P,3]` | float32 | neighborhood points in each contact frame |
| `normals_local` | `[P,3]` | float32 | face normals in each contact frame |
| `offsets` | `[C+1]` | int64 | flattened contact boundaries |
| `centers_camera` | `[C,3]` | float32 | centers in SAM camera axes |
| `normals_camera` | `[C,3]` | float32 | center face normals in SAM camera axes |
| `R_camera_from_local` | `[C,3,3]` | float32 | local-to-SAM-camera rotation |
| `center_point_ids` | `[C]` | int64 | center IDs in deterministic master pool |
| `center_face_ids` | `[C]` | int64 | refined-mesh containing face IDs |
| `center_barycentric` | `[C,3]` | float64 | center barycentric coordinates |
| `visibility` | `[C]` | int8 | center labels; currently all hidden (`0`) |
| `patch_radius` | `[C]` | float32 | geodesic radius per contact |
| `point_ids` | `[P]` | int64 | master-pool IDs; repeats across contacts are intentional |
| `point_visibility` | `[P]` | int8 | per-neighborhood-point labels `-1/0/1` |
| `keep_priority` | `[P]` | float64 | fixed master-point random priorities |
| `geodesic_distance` | `[P]` | float32 | distance to this contact center |
| `format_version` | scalar | int64 | current value `4` |
| `method` | scalar string | Unicode | current value `geodesic_ball` |
| `max_edge` | scalar | float64 | requested refinement threshold |
| `distance_method` | scalar string | Unicode | current value `fast_marching` |
| `method_args` | scalar JSON string | Unicode | fully resolved sampling parameters |
| `density` | scalar | int64 | master points per normalized area |
| `surface_point_count` | scalar | int64 | full object master-pool size |
| `surface_area` | scalar | float64 | refined normalized-object area |
| `surface_seed_parts` | `[2]` | int64 | object pool seed parts |
| `seed_parts` | `[3]` | int64 | per-view selection seed parts |

All fields load with `allow_pickle=False`.

## Real local sample audit

The only locally materialized object is:

```text
b7bc2efa5a784a05ab2be5a18d7da04c
```

It is assigned to the test split and has all 16 manifest rows and all view files.
There is no local source OBJ for it.

### Object-level touch pool

| Quantity | Value |
|---|---:|
| normalized surface area | 1.6043621716 |
| density | 200,000 |
| master-pool points | 320,873 |
| contacts per view | 32 |
| radius | 0.1 |
| refined max-edge setting | 0.03 |
| method | geodesic ball / fast marching / geodesic farthest |

### View-000 touch arrays

| Quantity | Value |
|---|---:|
| flattened points `P` | 170,069 |
| points/contact minimum | 4,251 |
| points/contact maximum | 5,676 |
| points/contact mean | 5,314.7 |
| hidden neighborhood entries | 149,917 |
| visible neighborhood entries | 15,152 |
| unknown neighborhood entries | 5,000 |

All 170,069 `point_ids` happen to be unique within this view, so these entry counts
also equal unique included-master-point counts here. The format still permits repeated
IDs when contact neighborhoods overlap; do not generalize this sample property.

Across 16 views, flattened `P` ranges approximately 164,388–170,428; global observed
contact sizes range approximately 4,095–5,955. The 16 files together store 2,689,032
flattened entries, including repeated master points across overlapping contacts/views.

`points_local` in view 000 spans approximately:

```text
min = (-0.0974, -0.0981, -0.0800)
max = ( 0.0982,  0.0977,  0.0588)
```

The Euclidean range need not equal the geodesic radius, and local Z need not be zero
away from the center.

Centers are not the same across views. This is expected from view-dependent hidden
eligibility and per-view seeds even though the master pool is shared.

## Full manifest snapshot

`generated_data/samples.jsonl` has 14,701 unique sample IDs representing 928 objects:

| Split | Samples | Represented objects |
|---|---:|---:|
| train | 11,713 | 739 |
| validation | 1,499 | 95 |
| test | 1,489 | 94 |

View-count distribution among represented objects:

- 915 objects have all 16 views;
- 5 have 1 view;
- 3 have 2 views;
- one object each has 7, 8, 9, 12, or 14 views.

All 14,701 inspected records have `target_path: null`.

After offline target generation, each represented object's rows should all contain
the same relative path:

```text
generated_data/<object_id>/target_latent.npz
```

No other manifest field or per-view artifact needs to change.

The failure log has 109 entries/unique object IDs:

- 91 `RuntimeError` records reporting that nonmanifold mesh handling was not
  implemented for the encountered topology;
- 18 `ValueError` records from excessive fragmentation.

Do not treat represented objects plus failure rows as a clean partition of the 1024
split IDs. The files are operational logs from append/resume behavior, and incomplete
or retried objects can reflect different points in a run.

## Manifest write/resume behavior

The current orchestrator appends successful records and failures rather than building
one atomic immutable dataset index. Completed sample IDs are read so already recorded
views can be skipped. Rendering jobs are managed per GPU and touch preparation uses a
process pool.

Loader and generation code should therefore tolerate:

- incomplete objects;
- files referred to by a copied manifest but absent on the current machine;
- failure logs that outlive retries;
- targets being absent while image/touch inputs are present;
- interrupted generation between file creation and manifest append.

Before training, build a validated immutable training index rather than consuming the
operational JSONL blindly. Validate every referenced path, file shape/dtype, split,
contact count, finite values, format version, and target compatibility.

## Loader implications

### Path resolution

Manifest paths such as
`generated_data/<id>/views/<view>/image.png` are relative to the data root, not to the
fork. The cluster and local roots differ. Keep the data root configurable and avoid
serializing absolute paths into model checkpoints.

### Variable point count

Contacts are fixed at 32 in the current files, but each contact has a variable number
of points. A contact encoder therefore needs an internal point mask or an exactly
fixed point-selection rule.

Useful deterministic choices include:

- threshold `keep_priority < fraction` for nested density experiments; this leaves
  variable counts and can theoretically produce empty contacts at very low fractions;
- take the `K` smallest priorities per contact for fixed-size batching; this is nested
  as K grows but corresponds to a point count rather than an exact density fraction;
- encode variable contacts independently and pool before batching contact tokens.

The first implementation should define behavior when a contact has fewer than `K`
points, including the padding mask used inside the touch encoder.

### Contact count and Stage 1 attention

The released Stage 1 cross-attention has no condition-token mask. Fixed 32 contacts
and a fixed number of output tokens per contact avoid ambiguous attention padding.
If contact-count ablations are needed, a learned null-contact token or an actual
attention-mask extension is safer than appending arbitrary zero tokens.

### Coordinate normalization

Raw `centers_camera` and reconstructed camera-space points match raw `pointmap.npy`.
At training/inference time they must undergo the same checkpoint-selected pointmap
scene normalization as the image pointmap. Do not invent an independent touch
normalization without an explicit reason and inverse mapping.

Local offsets are in normalized object units, while centers are in camera units. If
the pointmap normalizer applies a scene scale, local offsets must be scaled
consistently before combining them with normalized centers. Rotations/normals require
the mathematically appropriate handling if normalization is not a pure isotropic
scale plus translation. Reuse the instantiated normalizer rather than reimplementing
its assumed formula.

### Leakage and evaluation

Because current all-mode patches can include visible/unknown points, document which
point labels are fed to the model. At minimum, compare:

- all saved neighborhood points;
- hidden points only (`point_visibility == 0`);
- camera-visible points only as a diagnostic;
- center-only or center+orientation controls.

For a hidden-geometry claim, evaluate on target voxels/surfaces that are hidden from
the input camera and, ideally, separate regions inside and outside supplied touch
patches. Otherwise improvement may only show local copying of already visible or
directly supplied geometry.

## Data invariants worth unit-testing

1. Pointmap Z equals depth at every finite pixel.
2. Pointmap and touch use the same SAM camera axes.
3. For every contact, `offsets` are monotonic and bound all flattened fields equally.
4. `R.T @ R == I`, `det(R) == 1`, and `R[:,2] == normals_camera`.
5. Reconstructed camera points/normals match direct object-to-camera transforms.
6. Every center has hidden visibility and a zero-distance/zero-local-position point.
7. Distances are finite and at most `patch_radius` inside a contact.
8. `keep_priority` agrees for repeated `point_ids` across contacts and views.
9. `surface_point_count == ceil(surface_area * density)` for current generation.
10. Split assignment is consistent for all views of an object.
11. Every target mean has shape `[8,16,16,16]`, finite float values, and the expected
    encoder/config hash.
12. All views of one object reference the same target file.
13. Target voxel centers transformed by `T_Csam_from_O` align with the pointmap and
    touch geometry under an asymmetric fixture whose axis orientation is unmistakable.
