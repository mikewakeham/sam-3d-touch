# Where a coordinate correction can enter Stage 1

12 September 2026. Follow-up to the completed frame-factor experiment. No production changes or GPU job in this audit.

## Verified source contract

- `dataloader.py:146-174`: ordinary samples provide RGBA, pointmap, point coordinates and target. `camera.npz` is opened only for the oracle option, which supplies `object_from_camera`. Sample IDs are bookkeeping, not condition tokens.
- `train.py:258-290`: oracle geometry is transformed before encoding. The ordinary path normalizes camera geometry using pointmap SSI. Neither path adds a camera-to-object rotation token.
- `render_blender.py:95-134`: the target's object frame preserves source-asset orientation after a fixed import-axis conversion and centering/scaling. There is no geometry-based or semantic canonicalization step here. This does not establish that the source assets lack consistent semantic orientation.
- `render_blender.py:263-294` and `make_data.py:110`: view angles have an object-dependent random offset. A view index alone is not an object-frame pose measurement. Camera calibration into a robot/world frame would also need the relation between that frame and the target asset frame.
- `train.py:247-254,112-115`: non-shape flow targets are zero and their loss weights are zero. Existing layout parameters are not being supervised to predict these assets' camera-to-target poses.
- `modules/transformer/modulated.py:218-229` constructs multimodal self-attention with the default protected shape stream; `modules/attention/modules.py:194,309-335` makes shape attend only to shape. Layout can read detached shape, not the reverse. Condition cross-attention remains a separate route. Adding layout supervision alone therefore cannot create a layout-to-shape information path. This is a source-level dependency finding, not a measured accuracy diagnosis.

Consequently, using a predicted SAM layout rotation for point alignment would require an explicit additional pass/interface and validation against these target frames. It is not an existing correction that can be switched on. Unfreezing more shape weights also does not remove this attention separation, although those weights could learn orientation recovery from the observations themselves.

## Executed metadata check

`audit_pose_observability.py` reads the existing 64 samples from 32 objects and validates proper rotations. `pose_observability_results.json` records inputs/source hashes and per-sample results.

Object-origin translation in camera axes remains within roughly 8.4e-8 of zero in x/y and [1.99999952, 2] in z. Within-object relative rotations across the selected view pairs range from 30.31 to 180 degrees (median 113.86). Thus the renderer's camera translation is effectively constant while orientation changes substantially. It cannot stand in for the missing orientation. Camera position in object axes is a different quantity and requires the extrinsic rotation to calculate.

This audit does not establish that image/geometry observations cannot reveal pose. No actual observational collision or asset-wide canonicalization impossibility has been demonstrated.

## Consequences for the next intervention

Assume only camera-frame observations are available until the user specifies independent camera-to-target pose measurements.

1. If that rotation is measured, pre-encoder alignment is already the validated positive control. Confirm its exact frame convention rather than training a new encoder or repeating oracle full training. Its residual conditioning problem remains to solve.
2. Without measured rotation, a usable pre-encoder correction needs a learned alignment estimate (or a changed output-frame contract). Retain VecSetX. A learned alignment experiment must supervise the saved rotation explicitly, reserve both unseen views and unseen objects, and measure the resulting *geometric alignment*, not just angular error: shape symmetries may make distinct rotations equivalent. Its corrected input can then be evaluated against the existing known-pose and ordinary-camera endpoints. Failed finite-budget pose fitting alone would not prove unidentifiability.
3. The currently demonstrated upper bound for alignment alone is the oracle treatment, which still has residual view dependence and did not solve the user's dataset-wide result. Do not market a pose estimator as a sufficient full-surface fix. Before full training, the aligned model must show actual incremental use of correct surfaces over wrong-object/absent surfaces under varying visual inputs. Splitting the previously joint RGB+pointmap intervention is the remaining diagnostic for that residual.

Do not replace the encoder, infer object pose from camera translation, rotate an arbitrary latent as if its channels were xyz, or use target metadata silently at inference. No new full-dataset run follows from this audit.
